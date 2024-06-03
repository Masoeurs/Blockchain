from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from mssf_rs import G1Element

from mssf.types.blockchain_format.sized_bytes import bytes32
from mssf.util.byte_types import hexstr_to_bytes
from mssf.util.config import load_config, lock_and_load_config, save_config
from mssf.util.streamable import Streamable, streamable

"""
Config example
This is what goes into the user's config file, to communicate between the wallet and the farmer processes.
pool_list:
    launcher_id: xxx
    owner_public_key: xxx
    payout_instructions: xxx
    pool_url: xxx
    p2_singleton_puzzle_hash: xxx
    target_puzzle_hash: xxx
"""  # noqa

log = logging.getLogger(__name__)


@streamable
@dataclass(frozen=True)
class PoolWalletConfig(Streamable):
    launcher_id: bytes32
    pool_url: str
    payout_instructions: str
    target_puzzle_hash: bytes32
    p2_singleton_puzzle_hash: bytes32
    owner_public_key: G1Element


def load_pool_config(root_path: Path) -> List[PoolWalletConfig]:
    config = load_config(root_path, "config.yaml")
    ret_list: List[PoolWalletConfig] = []
    pool_list = config["pool"].get("pool_list", [])
    if pool_list is not None:
        for pool_config_dict in pool_list:
            try:
                pool_config = PoolWalletConfig(
                    bytes32.from_hexstr(pool_config_dict["launcher_id"]),
                    pool_config_dict["pool_url"],
                    pool_config_dict["payout_instructions"],
                    bytes32.from_hexstr(pool_config_dict["target_puzzle_hash"]),
                    bytes32.from_hexstr(pool_config_dict["p2_singleton_puzzle_hash"]),
                    G1Element.from_bytes(hexstr_to_bytes(pool_config_dict["owner_public_key"])),
                )
                ret_list.append(pool_config)
            except Exception as e:
                log.error(f"Exception loading config: {pool_config_dict} {e}")

    return ret_list


# TODO: remove this a few versions after 1.3, since authentication_public_key is deprecated. This is here to support
# downgrading to versions older than 1.3.
def add_auth_key(root_path: Path, pool_wallet_config: PoolWalletConfig, auth_key: G1Element) -> None:
    def update_auth_pub_key_for_entry(config_entry: Dict[str, Any]) -> bool:
        auth_key_hex = bytes(auth_key).hex()
        if config_entry.get("authentication_public_key", "") != auth_key_hex:
            config_entry["authentication_public_key"] = auth_key_hex

            return True

        return False

    update_pool_config_entry(
        root_path=root_path,
        pool_wallet_config=pool_wallet_config,
        update_closure=update_auth_pub_key_for_entry,
        update_log_message=f"Updating pool config for auth key: {auth_key}",
    )


def update_pool_url(root_path: Path, pool_wallet_config: PoolWalletConfig, pool_url: str) -> None:
    def update_pool_url_for_entry(config_entry: Dict[str, Any]) -> bool:
        if config_entry.get("pool_url", "") != pool_url:
            config_entry["pool_url"] = pool_url

            return True

        return False

    update_pool_config_entry(
        root_path=root_path,
        pool_wallet_config=pool_wallet_config,
        update_closure=update_pool_url_for_entry,
        update_log_message=f"Updating pool config for pool_url change: {pool_wallet_config.pool_url} -> {pool_url}",
    )


def update_pool_config_entry(
    root_path: Path,
    pool_wallet_config: PoolWalletConfig,
    update_closure: Callable[[Dict[str, Any]], bool],
    update_log_message: str,
) -> None:
    with lock_and_load_config(root_path, "config.yaml") as config:
        pool_list = config["pool"].get("pool_list", [])
        updated = False
        if pool_list is not None:
            for pool_config_dict in pool_list:
                try:
                    if hexstr_to_bytes(pool_config_dict["owner_public_key"]) == bytes(
                        pool_wallet_config.owner_public_key
                    ):
                        if update_closure(pool_config_dict):
                            updated = True
                except Exception as e:
                    log.error(f"Exception updating config: {pool_config_dict} {e}")
        if updated:
            log.info(update_log_message)
            config["pool"]["pool_list"] = pool_list
            save_config(root_path, "config.yaml", config)


async def update_pool_config(root_path: Path, pool_config_list: List[PoolWalletConfig]) -> None:
    with lock_and_load_config(root_path, "config.yaml") as full_config:
        full_config["pool"]["pool_list"] = [c.to_json_dict() for c in pool_config_list]
        save_config(root_path, "config.yaml", full_config)
