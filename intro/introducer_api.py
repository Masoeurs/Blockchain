from __future__ import annotations

import logging
from typing import Optional

from MSSF.introducer.introducer import Introducer
from MSSF.protocols.introducer_protocol import RequestPeersIntroducer, RespondPeersIntroducer
from MSSF.protocols.protocol_message_types import ProtocolMessageTypes
from MSSF.rpc.rpc_server import StateChangedProtocol
from MSSF.server.outbound_message import Message, make_msg
from MSSF.server.ws_connection import WSMSSFConnection
from MSSF.types.peer_info import TimestampedPeerInfo
from MSSF.util.api_decorators import api_request
from MSSF.util.ints import uint64


class IntroducerAPI:
    log: logging.Logger
    introducer: Introducer

    def __init__(self, introducer) -> None:
        self.log = logging.getLogger(__name__)
        self.introducer = introducer

    def ready(self) -> bool:
        return True

    def _set_state_changed_callback(self, callback: StateChangedProtocol) -> None:
        pass

    @api_request(peer_required=True)
    async def request_peers_introducer(
        self,
        request: RequestPeersIntroducer,
        peer: WSMSSFConnection,
    ) -> Optional[Message]:
        max_peers = self.introducer.max_peers_to_send
        if self.introducer.server is None or self.introducer.server.introducer_peers is None:
            return None
        rawpeers = self.introducer.server.introducer_peers.get_peers(
            max_peers * 5, True, self.introducer.recent_peer_threshold
        )

        peers = []
        for r_peer in rawpeers:
            if r_peer.vetted <= 0:
                continue

            if r_peer.host == peer.peer_info.host and r_peer.port == peer.peer_server_port:
                continue
            peer_without_timestamp = TimestampedPeerInfo(
                r_peer.host,
                r_peer.port,
                uint64(0),
            )
            peers.append(peer_without_timestamp)

            if len(peers) >= max_peers:
                break

        self.introducer.log.info(f"Sending vetted {peers}")

        msg = make_msg(ProtocolMessageTypes.respond_peers_introducer, RespondPeersIntroducer(peers))
        return msg
