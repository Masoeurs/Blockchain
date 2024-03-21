from __future__ import annotations

import asyncio
import os
import random
import sys
from pathlib import Path
from time import monotonic
from typing import List

from benchmarks.utils import setup_db
from MSSF.consensus.bl_record import blRecord
from MSSF.full_node.bl_store import blStore
from MSSF.types.blchain_format.foliage import Foliage, FoliageblData, FoliageTransactionbl, TransactionsInfo
from MSSF.types.blchain_format.pool_target import PoolTarget
from MSSF.types.blchain_format.proof_of_space import ProofOfSpace
from MSSF.types.blchain_format.reward_chain_bl import RewardChainbl
from MSSF.types.blchain_format.serialized_program import SerializedProgram
from MSSF.types.blchain_format.sized_bytes import bytes32
from MSSF.types.blchain_format.sub_epoch_summary import SubEpochSummary
from MSSF.types.full_bl import Fullbl
from MSSF.util.ints import uint8, uint32, uint64, uint128
from tests.util.benchmarks import (
    clvm_generator,
    rand_bytes,
    rand_class_group_element,
    rand_g1,
    rand_g2,
    rand_hash,
    rand_vdf,
    rand_vdf_proof,
    rewards,
)

# to run this benchmark:
# python -m benchmarks.coin_store

NUM_ITERS = 20000

# we need seeded random, to have reproducible benchmark runs
random.seed(123456789)


async def run_add_bl_benchmark(version: int) -> None:
    verbose: bool = "--verbose" in sys.argv

    # keep track of benchmark total time
    all_test_time = 0.0

    prev_bl = bytes32([0] * 32)
    prev_ses_hash = bytes32([0] * 32)

    header_hashes = []

    async with setup_db("bl-store-benchmark.db", version) as db_wrapper:
        bl_store = await blStore.create(db_wrapper)

        bl_height = 1
        timestamp = uint64(1631794488)
        weight = uint128(10)
        iters = uint128(123456)
        sp_index = uint8(0)
        deficit = uint8(0)
        sub_slot_iters = uint64(10)
        required_iters = uint64(100)
        transaction_bl_counter = 0
        prev_transaction_bl = bytes32([0] * 32)
        prev_transaction_height = uint32(0)
        total_time = 0.0
        ses_counter = 0

        if verbose:
            print("profiling add_full_bl", end="")

        tx_bl_heights: List[uint32] = []

        for height in range(bl_height, bl_height + NUM_ITERS):
            is_transaction = transaction_bl_counter == 0
            if is_transaction:
                tx_bl_heights.append(uint32(height))
            fees = uint64(random.randint(0, 150000))
            farmer_coin, pool_coin = rewards(uint32(height))
            reward_claims_incorporated = [farmer_coin, pool_coin]

            # TODO: increase fidelity by setting these as well
            finished_challenge_slot_hashes = None
            finished_infused_challenge_slot_hashes = None
            finished_reward_slot_hashes = None

            sub_epoch_summary_included = None
            if ses_counter == 0:
                sub_epoch_summary_included = SubEpochSummary(
                    prev_ses_hash,
                    rand_hash(),
                    uint8(random.randint(0, 255)),  # num_bls_overflow: uint8
                    None,  # new_difficulty: Optional[uint64]
                    None,  # new_sub_slot_iters: Optional[uint64]
                )

            has_pool_pk = random.randint(0, 1)

            proof_of_space = ProofOfSpace(
                rand_hash(),  # challenge
                rand_g1() if has_pool_pk else None,
                rand_hash() if not has_pool_pk else None,
                rand_g1(),  # plot_public_key
                uint8(32),
                rand_bytes(8 * 32),
            )

            reward_chain_bl = RewardChainbl(
                weight,
                uint32(height),
                iters,
                sp_index,
                rand_hash(),  # pos_ss_cc_challenge_hash
                proof_of_space,
                None if sp_index == 0 else rand_vdf(),
                rand_g2(),  # challenge_chain_sp_signature
                rand_vdf(),  # challenge_chain_ip_vdf
                rand_vdf() if sp_index != 0 else None,  # reward_chain_sp_vdf
                rand_g2(),  # reward_chain_sp_signature
                rand_vdf(),  # reward_chain_ip_vdf
                rand_vdf() if deficit < 16 else None,
                is_transaction,
            )

            pool_target = PoolTarget(
                rand_hash(),  # puzzle_hash
                uint32(0),  # max_height
            )

            foliage_bl_data = FoliageblData(
                rand_hash(),  # unfinished_reward_bl_hash
                pool_target,
                rand_g2() if has_pool_pk else None,  # pool_signature
                rand_hash(),  # farmer_reward_puzzle_hash
                bytes32([0] * 32),  # extension_data
            )

            foliage = Foliage(
                prev_bl,
                rand_hash(),  # reward_bl_hash
                foliage_bl_data,
                rand_g2(),  # foliage_bl_data_signature
                rand_hash() if is_transaction else None,  # foliage_transaction_bl_hash
                rand_g2() if is_transaction else None,  # foliage_transaction_bl_signature
            )

            foliage_transaction_bl = (
                None
                if not is_transaction
                else FoliageTransactionbl(
                    prev_transaction_bl,
                    timestamp,
                    rand_hash(),  # filter_hash
                    rand_hash(),  # additions_root
                    rand_hash(),  # removals_root
                    rand_hash(),  # transactions_info_hash
                )
            )

            transactions_info = (
                None
                if not is_transaction
                else TransactionsInfo(
                    rand_hash(),  # generator_root
                    rand_hash(),  # generator_refs_root
                    rand_g2(),  # aggregated_signature
                    fees,
                    uint64(random.randint(0, 12000000000)),  # cost
                    reward_claims_incorporated,
                )
            )

            full_bl = Fullbl(
                [],  # finished_sub_slots
                reward_chain_bl,
                rand_vdf_proof() if sp_index > 0 else None,  # challenge_chain_sp_proof
                rand_vdf_proof(),  # challenge_chain_ip_proof
                rand_vdf_proof() if sp_index > 0 else None,  # reward_chain_sp_proof
                rand_vdf_proof(),  # reward_chain_ip_proof
                rand_vdf_proof() if deficit < 4 else None,  # infused_challenge_chain_ip_proof
                foliage,
                foliage_transaction_bl,
                transactions_info,
                SerializedProgram.from_bytes(clvm_generator) if is_transaction else None,  # transactions_generator
                [],  # transactions_generator_ref_list
            )

            header_hash = full_bl.header_hash

            record = blRecord(
                header_hash,
                prev_bl,
                uint32(height),
                weight,
                iters,
                sp_index,
                rand_class_group_element(),
                None if deficit > 3 else rand_class_group_element(),
                rand_hash(),  # reward_infusion_new_challenge
                rand_hash(),  # challenge_bl_info_hash
                sub_slot_iters,
                rand_hash(),  # pool_puzzle_hash
                rand_hash(),  # farmer_puzzle_hash
                required_iters,
                deficit,
                deficit == 16,
                prev_transaction_height,
                timestamp if is_transaction else None,
                prev_transaction_bl if prev_transaction_bl != bytes32([0] * 32) else None,
                None if fees == 0 else fees,
                reward_claims_incorporated,
                finished_challenge_slot_hashes,
                finished_infused_challenge_slot_hashes,
                finished_reward_slot_hashes,
                sub_epoch_summary_included,
            )

            start = monotonic()
            await bl_store.add_full_bl(header_hash, full_bl, record)
            await bl_store.set_in_chain([(header_hash,)])
            header_hashes.append(header_hash)
            await bl_store.set_peak(header_hash)

            stop = monotonic()
            total_time += stop - start

            # 19 seconds per bl
            timestamp = uint64(timestamp + 19)
            weight = uint128(weight + 10)
            iters = uint128(iters + 123456)
            sp_index = uint8((sp_index + 1) % 64)
            deficit = uint8((deficit + 3) % 17)
            ses_counter = (ses_counter + 1) % 384
            prev_bl = header_hash

            # every 33 bls is a transaction bl
            transaction_bl_counter = (transaction_bl_counter + 1) % 33

            if is_transaction:
                prev_transaction_bl = header_hash
                prev_transaction_height = uint32(height)

            if ses_counter == 0:
                prev_ses_hash = header_hash

            if verbose:
                print(".", end="")
                sys.stdout.flush()
        bl_height += NUM_ITERS

        if verbose:
            print("")
        print(f"{total_time:0.4f}s, add_full_bl")
        all_test_time += total_time

        # === get_bl_info() ===
        total_time = 0.0
        if verbose:
            print("profiling get_bl_info")

        random.shuffle(header_hashes)
        start = monotonic()
        for h in header_hashes:
            bl_info = await bl_store.get_bl_info(h)
            assert bl_info is not None
            assert len(bl_info.prev_header_hash) == 32

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_bl_info")
        all_test_time += total_time

        # === get_generator() ===
        total_time = 0.0
        if verbose:
            print("profiling get_generator")

        random.shuffle(header_hashes)
        start = monotonic()
        for h in header_hashes:
            await bl_store.get_generator(h)

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_generator")
        all_test_time += total_time

        # === get_full_bl() ===
        total_time = 0.0
        if verbose:
            print("profiling get_full_bl")

        random.shuffle(header_hashes)
        start = monotonic()
        for h in header_hashes:
            bl = await bl_store.get_full_bl(h)
            assert bl is not None
            assert bl.header_hash == h

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_full_bl")
        all_test_time += total_time

        # === get_full_bl_bytes() ===
        total_time = 0.0
        if verbose:
            print("profiling get_full_bl_bytes")

        start = monotonic()
        for h in header_hashes:
            bl_bs = await bl_store.get_full_bl_bytes(h)
            assert bl_bs is not None
            assert len(bl_bs) > 0

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_full_bl_bytes")
        all_test_time += total_time

        # === get_generators_at() ===
        total_time = 0.0
        if verbose:
            print("profiling get_generators_at")

        start = monotonic()
        for i in tx_bl_heights:
            gens = await bl_store.get_generators_at([i])
            assert len(gens) == 1

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_generators_at")
        all_test_time += total_time

        # === get_full_bls_at() ===
        total_time = 0.0
        if verbose:
            print("profiling get_full_bls_at")

        start = monotonic()
        for hi in range(1, bl_height):
            bls = await bl_store.get_full_bls_at([uint32(hi)])
            assert len(bls) == 1
            assert bls[0].height == hi

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_full_bls_at")
        all_test_time += total_time

        # === get_bl_records_by_hash() ===
        total_time = 0.0
        if verbose:
            print("profiling get_bl_records_by_hash")

        start = monotonic()
        for h in header_hashes:
            bl_recs = await bl_store.get_bl_records_by_hash([h])
            assert len(bl_recs) == 1
            assert bl_recs[0].header_hash == h

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_bl_records_by_hash")
        all_test_time += total_time

        # === get_bl_bytes_by_hash() ===
        total_time = 0.0
        if verbose:
            print("profiling get_bl_bytes_by_hash")

        start = monotonic()
        for h in header_hashes:
            blobs = await bl_store.get_bl_bytes_by_hash([h])
            assert len(blobs) == 1

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_bl_bytes_by_hash")
        all_test_time += total_time

        # === get_bls_by_hash() ===
        total_time = 0.0
        if verbose:
            print("profiling get_bls_by_hash")

        start = monotonic()
        for h in header_hashes:
            bls = await bl_store.get_bls_by_hash([h])
            assert len(bls) == 1
            assert bls[0].header_hash == h

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_bls_by_hash")
        all_test_time += total_time

        # === get_bl_record() ===
        total_time = 0.0
        if verbose:
            print("profiling get_bl_record")

        start = monotonic()
        for h in header_hashes:
            bl_rec = await bl_store.get_bl_record(h)
            assert bl_rec is not None
            assert bl_rec.header_hash == h

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_bl_record")
        all_test_time += total_time

        # === get_bl_records_in_range() ===
        total_time = 0.0
        if verbose:
            print("profiling get_bl_records_in_range")

        start = monotonic()
        for _ in range(100):
            hi = random.randint(1, bl_height - 100)
            bls_dict = await bl_store.get_bl_records_in_range(hi, hi + 99)
            assert len(bls_dict) == 100

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_bl_records_in_range")
        all_test_time += total_time

        # === get_bl_records_close_to_peak() ===
        total_time = 0.0
        if verbose:
            print("profiling get_bl_records_close_to_peak")

        start = monotonic()
        bl_dict, peak_h = await bl_store.get_bl_records_close_to_peak(99)
        assert len(bl_dict) == 100

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_bl_records_close_to_peak")
        all_test_time += total_time

        # === is_fully_compactified() ===
        total_time = 0.0
        if verbose:
            print("profiling is_fully_compactified")

        start = monotonic()
        for h in header_hashes:
            compactified = await bl_store.is_fully_compactified(h)
            assert compactified is False

        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, is_fully_compactified")
        all_test_time += total_time

        # === get_random_not_compactified() ===
        total_time = 0.0
        if verbose:
            print("profiling get_random_not_compactified")

        start = monotonic()
        for _ in range(1, 5000):
            bls_int_list = await bl_store.get_random_not_compactified(100)
            assert len(bls_int_list) == 100
        stop = monotonic()
        total_time += stop - start

        print(f"{total_time:0.4f}s, get_random_not_compactified")
        all_test_time += total_time

        print(f"all tests completed in {all_test_time:0.4f}s")

        db_size = os.path.getsize(Path("bl-store-benchmark.db"))
        print(f"database size: {db_size/1000000:.3f} MB")


if __name__ == "__main__":
    print("version 2")
    asyncio.run(run_add_bl_benchmark(2))
