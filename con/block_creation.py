from __future__ import annotations

import logging
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import MSSF_rs
from MSSF_rs import G1Element, G2Element, compute_merkle_set_root
from MSSFbip158 import PyBIP158


from MSSF.types.full_bm import Fullbm
from MSSF.types.generator_types import bmGenerator
from MSSF.types.unfinished_bm import Unfinishedbm
from MSSF.util.hash import std_hash
from MSSF.util.ints import uint8, uint32, uint64, uint128
from MSSF.util.prev_transaction_bm import get_prev_transaction_bm
from MSSF.consensus.bm_record import bmRecord
from MSSF.consensus.bm_rewards import calculate_base_farmer_reward, calculate_pool_reward
from MSSF.consensus.bmchain_interface import bmchainInterface
from MSSF.consensus.coinbase import create_farmer_coin, create_pool_coin
from MSSF.consensus.constants import ConsensusConstants
from MSSF.consensus.cost_calculator import NPCResult
from MSSF.full_node.mempool_check_conditions import get_name_puzzle_conditions
from MSSF.full_node.signage_point import SignagePoint
from MSSF.types.bmchain_format.coin import Coin, hash_coin_ids
from MSSF.types.bmchain_format.foliage import Foliage, FoliagebmData, FoliageTransactionbm, TransactionsInfo
from MSSF.types.bmchain_format.pool_target import PoolTarget
from MSSF.types.bmchain_format.proof_of_space import ProofOfSpace
from MSSF.types.bmchain_format.reward_chain_bm import RewardChainbm, RewardChainbmUnfinished
from MSSF.types.bmchain_format.sized_bytes import bytes32
from MSSF.types.bmchain_format.vdf import VDFInfo, VDFProof
from MSSF.types.end_of_slot_bundle import EndOfSubSlotBundle

log = logging.getLogger(__name__)


def compute_bm_cost(generator: bmGenerator, constants: ConsensusConstants, height: uint32) -> uint64:
    result: NPCResult = get_name_puzzle_conditions(
        generator, constants.MAX_bm_COST_CLVM, mempool_mode=True, height=height, constants=constants
    )
    return uint64(0 if result.conds is None else result.conds.cost)


def compute_bm_fee(additions: Sequence[Coin], removals: Sequence[Coin]) -> uint64:
    removal_amount = 0
    addition_amount = 0
    for coin in removals:
        removal_amount += coin.amount
    for coin in additions:
        addition_amount += coin.amount
    return uint64(removal_amount - addition_amount)


def create_foliage(
    constants: ConsensusConstants,
    reward_bm_unfinished: RewardChainbmUnfinished,
    bm_generator: Optional[bmGenerator],
    aggregate_sig: G2Element,
    bms: bmchainInterface,
    total_iters_sp: uint128,
    timestamp: uint64,
    additions: List[Coin],
    removals: List[Coin],
    prev_bm: Optional[bmRecord],
    farmer_reward_puzzlehash: bytes32,
    pool_target: PoolTarget,
    get_plot_signature: Callable[[bytes32, G1Element], G2Element],
    get_pool_signature: Callable[[PoolTarget, Optional[G1Element]], Optional[G2Element]],
    seed: bytes,
    compute_cost: Callable[[bmGenerator, ConsensusConstants, uint32], uint64],
    compute_fees: Callable[[Sequence[Coin], Sequence[Coin]], uint64],
) -> Tuple[Foliage, Optional[FoliageTransactionbm], Optional[TransactionsInfo]]:

    if prev_bm is not None:
        res = get_prev_transaction_bm(prev_bm, bms, total_iters_sp)
        is_transaction_bm: bool = res[0]
        prev_transaction_bm: Optional[bmRecord] = res[1]
    else:
        # Genesis is a transaction bm
        prev_transaction_bm = None
        is_transaction_bm = True

    rng = random.Random()
    rng.seed(seed)
    # Use the extension data to create different bms based on header hash
    extension_data: bytes32 = bytes32(rng.randint(0, 100000000).to_bytes(32, "big"))
    if prev_bm is None:
        height: uint32 = uint32(0)
    else:
        height = uint32(prev_bm.height + 1)

    # Create filter
    byte_array_tx: List[bytearray] = []
    tx_additions: List[Coin] = []
    tx_removals: List[bytes32] = []

    pool_target_signature: Optional[G2Element] = get_pool_signature(
        pool_target, reward_bm_unfinished.proof_of_space.pool_public_key
    )

    foliage_data = FoliagebmData(
        reward_bm_unfinished.get_hash(),
        pool_target,
        pool_target_signature,
        farmer_reward_puzzlehash,
        extension_data,
    )

    foliage_bm_data_signature: G2Element = get_plot_signature(
        foliage_data.get_hash(),
        reward_bm_unfinished.proof_of_space.plot_public_key,
    )

    prev_bm_hash: bytes32 = constants.GENESIS_CHALLENGE
    if height != 0:
        assert prev_bm is not None
        prev_bm_hash = prev_bm.header_hash

    generator_bm_heights_list: List[uint32] = []

    foliage_transaction_bm_hash: Optional[bytes32]

    if is_transaction_bm:
        cost = uint64(0)

        # Calculate the cost of transactions
        if bm_generator is not None:
            generator_bm_heights_list = bm_generator.bm_height_list
            cost = compute_cost(bm_generator, constants, height)

            spend_bundle_fees = compute_fees(additions, removals)
        else:
            spend_bundle_fees = uint64(0)

        reward_claims_incorporated = []
        if height > 0:
            assert prev_transaction_bm is not None
            assert prev_bm is not None
            curr: bmRecord = prev_bm
            while not curr.is_transaction_bm:
                curr = bms.bm_record(curr.prev_hash)

            assert curr.fees is not None
            pool_coin = create_pool_coin(
                curr.height, curr.pool_puzzle_hash, calculate_pool_reward(curr.height), constants.GENESIS_CHALLENGE
            )

            farmer_coin = create_farmer_coin(
                curr.height,
                curr.farmer_puzzle_hash,
                uint64(calculate_base_farmer_reward(curr.height) + curr.fees),
                constants.GENESIS_CHALLENGE,
            )
            assert curr.header_hash == prev_transaction_bm.header_hash
            reward_claims_incorporated += [pool_coin, farmer_coin]

            if curr.height > 0:
                curr = bms.bm_record(curr.prev_hash)
                # Prev bm is not genesis
                while not curr.is_transaction_bm:
                    pool_coin = create_pool_coin(
                        curr.height,
                        curr.pool_puzzle_hash,
                        calculate_pool_reward(curr.height),
                        constants.GENESIS_CHALLENGE,
                    )
                    farmer_coin = create_farmer_coin(
                        curr.height,
                        curr.farmer_puzzle_hash,
                        calculate_base_farmer_reward(curr.height),
                        constants.GENESIS_CHALLENGE,
                    )
                    reward_claims_incorporated += [pool_coin, farmer_coin]
                    curr = bms.bm_record(curr.prev_hash)
        additions.extend(reward_claims_incorporated.copy())
        for coin in additions:
            tx_additions.append(coin)
            byte_array_tx.append(bytearray(coin.puzzle_hash))
        for coin in removals:
            cname = coin.name()
            tx_removals.append(cname)
            byte_array_tx.append(bytearray(cname))

        bip158: PyBIP158 = PyBIP158(byte_array_tx)
        encoded = bytes(bip158.GetEncoded())

        additions_merkle_items: List[bytes32] = []

        # Create addition Merkle set
        puzzlehash_coin_map: Dict[bytes32, List[bytes32]] = {}

        for coin in tx_additions:
            if coin.puzzle_hash in puzzlehash_coin_map:
                puzzlehash_coin_map[coin.puzzle_hash].append(coin.name())
            else:
                puzzlehash_coin_map[coin.puzzle_hash] = [coin.name()]

        # Addition Merkle set contains puzzlehash and hash of all coins with that puzzlehash
        for puzzle, coin_ids in puzzlehash_coin_map.items():
            additions_merkle_items.append(puzzle)
            additions_merkle_items.append(hash_coin_ids(coin_ids))

        additions_root = bytes32(compute_merkle_set_root(additions_merkle_items))
        removals_root = bytes32(compute_merkle_set_root(tx_removals))

        generator_hash = bytes32([0] * 32)
        if bm_generator is not None:
            generator_hash = std_hash(bm_generator.program)

        generator_refs_hash = bytes32([1] * 32)
        if generator_bm_heights_list not in (None, []):
            generator_ref_list_bytes = b"".join([i.stream_to_bytes() for i in generator_bm_heights_list])
            generator_refs_hash = std_hash(generator_ref_list_bytes)

        filter_hash: bytes32 = std_hash(encoded)

        transactions_info: Optional[TransactionsInfo] = TransactionsInfo(
            generator_hash,
            generator_refs_hash,
            aggregate_sig,
            spend_bundle_fees,
            cost,
            reward_claims_incorporated,
        )
        if prev_transaction_bm is None:
            prev_transaction_bm_hash: bytes32 = constants.GENESIS_CHALLENGE
        else:
            prev_transaction_bm_hash = prev_transaction_bm.header_hash

        assert transactions_info is not None
        foliage_transaction_bm: Optional[FoliageTransactionbm] = FoliageTransactionbm(
            prev_transaction_bm_hash,
            timestamp,
            filter_hash,
            additions_root,
            removals_root,
            transactions_info.get_hash(),
        )
        assert foliage_transaction_bm is not None

        foliage_transaction_bm_hash = foliage_transaction_bm.get_hash()
        foliage_transaction_bm_signature: Optional[G2Element] = get_plot_signature(
            foliage_transaction_bm_hash, reward_bm_unfinished.proof_of_space.plot_public_key
        )
        assert foliage_transaction_bm_signature is not None
    else:
        foliage_transaction_bm_hash = None
        foliage_transaction_bm_signature = None
        foliage_transaction_bm = None
        transactions_info = None
    assert (foliage_transaction_bm_hash is None) == (foliage_transaction_bm_signature is None)

    foliage = Foliage(
        prev_bm_hash,
        reward_bm_unfinished.get_hash(),
        foliage_data,
        foliage_bm_data_signature,
        foliage_transaction_bm_hash,
        foliage_transaction_bm_signature,
    )

    return foliage, foliage_transaction_bm, transactions_info


def create_unfinished_bm(
    constants: ConsensusConstants,
    sub_slot_start_total_iters: uint128,
    sub_slot_iters: uint64,
    signage_point_index: uint8,
    sp_iters: uint64,
    ip_iters: uint64,
    proof_of_space: ProofOfSpace,
    slot_cc_challenge: bytes32,
    farmer_reward_puzzle_hash: bytes32,
    pool_target: PoolTarget,
    get_plot_signature: Callable[[bytes32, G1Element], G2Element],
    get_pool_signature: Callable[[PoolTarget, Optional[G1Element]], Optional[G2Element]],
    signage_point: SignagePoint,
    timestamp: uint64,
    bms: bmchainInterface,
    seed: bytes = b"",
    bm_generator: Optional[bmGenerator] = None,
    aggregate_sig: G2Element = G2Element(),
    additions: Optional[List[Coin]] = None,
    removals: Optional[List[Coin]] = None,
    prev_bm: Optional[bmRecord] = None,
    finished_sub_slots_input: Optional[List[EndOfSubSlotBundle]] = None,
    compute_cost: Callable[[bmGenerator, ConsensusConstants, uint32], uint64] = compute_bm_cost,
    compute_fees: Callable[[Sequence[Coin], Sequence[Coin]], uint64] = compute_bm_fee,
) -> Unfinishedbm:
  
    if finished_sub_slots_input is None:
        finished_sub_slots: List[EndOfSubSlotBundle] = []
    else:
        finished_sub_slots = finished_sub_slots_input.copy()
    overflow: bool = sp_iters > ip_iters
    total_iters_sp: uint128 = uint128(sub_slot_start_total_iters + sp_iters)
    is_genesis: bool = prev_bm is None

    new_sub_slot: bool = len(finished_sub_slots) > 0

    cc_sp_hash: bytes32 = slot_cc_challenge

    # Only enters this if statement if we are in testing mode (making VDF proofs here)
    if signage_point.cc_vdf is not None:
        assert signage_point.rc_vdf is not None
        cc_sp_hash = signage_point.cc_vdf.output.get_hash()
        rc_sp_hash = signage_point.rc_vdf.output.get_hash()
    else:
        if new_sub_slot:
            rc_sp_hash = finished_sub_slots[-1].reward_chain.get_hash()
        else:
            if is_genesis:
                rc_sp_hash = constants.GENESIS_CHALLENGE
            else:
                assert prev_bm is not None
                assert bms is not None
                curr = prev_bm
                while not curr.first_in_sub_slot:
                    curr = bms.bm_record(curr.prev_hash)
                assert curr.finished_reward_slot_hashes is not None
                rc_sp_hash = curr.finished_reward_slot_hashes[-1]
        signage_point = SignagePoint(None, None, None, None)

    cc_sp_signature: Optional[G2Element] = get_plot_signature(cc_sp_hash, proof_of_space.plot_public_key)
    rc_sp_signature: Optional[G2Element] = get_plot_signature(rc_sp_hash, proof_of_space.plot_public_key)
    assert cc_sp_signature is not None
    assert rc_sp_signature is not None
    assert MSSF_rs.AugSchemeMPL.verify(proof_of_space.plot_public_key, cc_sp_hash, cc_sp_signature)

    total_iters = uint128(sub_slot_start_total_iters + ip_iters + (sub_slot_iters if overflow else 0))

    rc_bm = RewardChainbmUnfinished(
        total_iters,
        signage_point_index,
        slot_cc_challenge,
        proof_of_space,
        signage_point.cc_vdf,
        cc_sp_signature,
        signage_point.rc_vdf,
        rc_sp_signature,
    )
    if additions is None:
        additions = []
    if removals is None:
        removals = []
    (foliage, foliage_transaction_bm, transactions_info) = create_foliage(
        constants,
        rc_bm,
        bm_generator,
        aggregate_sig,
        additions,
        removals,
        prev_bm,
        bms,
        total_iters_sp,
        timestamp,
        farmer_reward_puzzle_hash,
        pool_target,
        get_plot_signature,
        get_pool_signature,
        seed,
        compute_cost,
        compute_fees,
    )
    return Unfinishedbm(
        finished_sub_slots,
        rc_bm,
        signage_point.cc_proof,
        signage_point.rc_proof,
        foliage,
        foliage_transaction_bm,
        transactions_info,
        bm_generator.program if bm_generator else None,
        bm_generator.bm_height_list if bm_generator else [],
    )


def unfinished_bm_to_full_bm(
    unfinished_bm: Unfinishedbm,
    cc_ip_vdf: VDFInfo,
    cc_ip_proof: VDFProof,
    rc_ip_vdf: VDFInfo,
    rc_ip_proof: VDFProof,
    icc_ip_vdf: Optional[VDFInfo],
    icc_ip_proof: Optional[VDFProof],
    finished_sub_slots: List[EndOfSubSlotBundle],
    prev_bm: Optional[bmRecord],
    bms: bmchainInterface,
    total_iters_sp: uint128,
    difficulty: uint64,
) -> Fullbm:
  
    # Replace things that need to be replaced, since foliage bms did not necessarily have the latest information
    if prev_bm is None:
        is_transaction_bm = True
        new_weight = uint128(difficulty)
        new_height = uint32(0)
        new_foliage_transaction_bm = unfinished_bm.foliage_transaction_bm
        new_tx_info = unfinished_bm.transactions_info
        new_generator = unfinished_bm.transactions_generator
        new_generator_ref_list = unfinished_bm.transactions_generator_ref_list
    else:
        is_transaction_bm, _ = get_prev_transaction_bm(prev_bm, bms, total_iters_sp)
        new_weight = uint128(prev_bm.weight + difficulty)
        new_height = uint32(prev_bm.height + 1)
        if is_transaction_bm:
            new_foliage_transaction_bm = unfinished_bm.foliage_transaction_bm
            new_tx_info = unfinished_bm.transactions_info
            new_generator = unfinished_bm.transactions_generator
            new_generator_ref_list = unfinished_bm.transactions_generator_ref_list
        else:
            new_foliage_transaction_bm = None
            new_tx_info = None
            new_generator = None
            new_generator_ref_list = []
    reward_chain_bm = RewardChainbm(
        new_weight,
        new_height,
        unfinished_bm.reward_chain_bm.total_iters,
        unfinished_bm.reward_chain_bm.signage_point_index,
        unfinished_bm.reward_chain_bm.pos_ss_cc_challenge_hash,
        unfinished_bm.reward_chain_bm.proof_of_space,
        unfinished_bm.reward_chain_bm.challenge_chain_sp_vdf,
        unfinished_bm.reward_chain_bm.challenge_chain_sp_signature,
        cc_ip_vdf,
        unfinished_bm.reward_chain_bm.reward_chain_sp_vdf,
        unfinished_bm.reward_chain_bm.reward_chain_sp_signature,
        rc_ip_vdf,
        icc_ip_vdf,
        is_transaction_bm,
    )
    if prev_bm is None:
        new_foliage = unfinished_bm.foliage.replace(reward_bm_hash=reward_chain_bm.get_hash())
    else:
        if is_transaction_bm:
            new_fbh = unfinished_bm.foliage.foliage_transaction_bm_hash
            new_fbs = unfinished_bm.foliage.foliage_transaction_bm_signature
        else:
            new_fbh = None
            new_fbs = None
        assert (new_fbh is None) == (new_fbs is None)
        new_foliage = unfinished_bm.foliage.replace(
            reward_bm_hash=reward_chain_bm.get_hash(),
            prev_bm_hash=prev_bm.header_hash,
            foliage_transaction_bm_hash=new_fbh,
            foliage_transaction_bm_signature=new_fbs,
        )
    ret = Fullbm(
        finished_sub_slots,
        reward_chain_bm,
        unfinished_bm.challenge_chain_sp_proof,
        cc_ip_proof,
        unfinished_bm.reward_chain_sp_proof,
        rc_ip_proof,
        icc_ip_proof,
        new_foliage,
        new_foliage_transaction_bm,
        new_tx_info,
        new_generator,
        new_generator_ref_list,
    )
    return ret
