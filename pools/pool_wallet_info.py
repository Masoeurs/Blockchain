from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional

from mssf_rs import G1Element

from mssf.protocols.pool_protocol import POOL_PROTOCOL_VERSION
from mssf.types.blockchain_format.coin import Coin
from mssf.types.blockchain_format.program import Program
from mssf.types.blockchain_format.sized_bytes import bytes32
from mssf.util.byte_types import hexstr_to_bytes
from mssf.util.ints import uint8, uint32
from mssf.util.streamable import Streamable, streamable


class PoolSingletonState(IntEnum):
    """
    From the user's point of view, a pool group can be in these states:
    `SELF_POOLING`: The singleton exists on the blockchain, and we are farming
        block rewards to a wallet address controlled by the user

    `LEAVING_POOL`: The singleton exists, and we have entered the "escaping" state, which
        means we are waiting for a number of blocks = `relative_lock_height` to pass, so we can leave.

    `FARMING_TO_POOL`: The singleton exists, and it is assigned to a pool.

    `CLAIMING_SELF_POOLED_REWARDS`: We have submitted a transaction to sweep our
        self-pooled funds.
    """

    SELF_POOLING = 1
    LEAVING_POOL = 2
    FARMING_TO_POOL = 3


SELF_POOLING = PoolSingletonState.SELF_POOLING
LEAVING_POOL = PoolSingletonState.LEAVING_POOL
FARMING_TO_POOL = PoolSingletonState.FARMING_TO_POOL


@streamable
@dataclass(frozen=True)
class PoolState(Streamable):
   

    version: uint8
    state: uint8  # PoolSingletonState
   
    target_puzzle_hash: bytes32 
    owner_pubkey: G1Element
    pool_url: Optional[str]
    relative_lock_height: uint32


def initial_pool_state_from_dict(
    state_dict: Dict[str, Any],
    owner_pubkey: G1Element,
    owner_puzzle_hash: bytes32,
) -> PoolState:
    state_str = state_dict["state"]
    singleton_state: PoolSingletonState = PoolSingletonState[state_str]

    if singleton_state == SELF_POOLING:
        target_puzzle_hash = owner_puzzle_hash
        pool_url: str = ""
        relative_lock_height = uint32(0)
    elif singleton_state == FARMING_TO_POOL:
        target_puzzle_hash = bytes32(hexstr_to_bytes(state_dict["target_puzzle_hash"]))
        pool_url = state_dict["pool_url"]
        relative_lock_height = uint32(state_dict["relative_lock_height"])
    else:
        raise ValueError("Initial state must be SELF_POOLING or FARMING_TO_POOL")

    # TODO: change create_pool_state to return error messages, as well
    assert relative_lock_height is not None
    return create_pool_state(singleton_state, target_puzzle_hash, owner_pubkey, pool_url, relative_lock_height)


def create_pool_state(
    state: PoolSingletonState,
    target_puzzle_hash: bytes32,
    owner_pubkey: G1Element,
    pool_url: Optional[str],
    relative_lock_height: uint32,
) -> PoolState:
    if state not in {s.value for s in PoolSingletonState}:
        raise AssertionError("state {state} is not a valid PoolSingletonState,")
    ps = PoolState(
        POOL_PROTOCOL_VERSION, uint8(state), target_puzzle_hash, owner_pubkey, pool_url, relative_lock_height
    )
    # TODO Move verify here
    return ps


@streamable
@dataclass(frozen=True)
class PoolWalletInfo(Streamable):
   
    current: PoolState
    target: Optional[PoolState]
    launcher_coin: Coin
    launcher_id: bytes32
    p2_singleton_puzzle_hash: bytes32
    current_inner: Program 
    tip_singleton_coin_id: bytes32
    singleton_block_height: uint32 
