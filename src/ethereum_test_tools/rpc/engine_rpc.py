"""
Ethereum `engine_X` JSON-RPC Engine API methods used within EEST based hive simulators.
"""

from typing import Dict, Union

from ..consume.engine.types import EngineCancun, EngineParis, EngineShanghai
from .base_rpc import BaseRPC

ForkchoiceStateV1 = Dict
PayloadAttributes = Dict


class EngineRPC(BaseRPC):
    """
    Represents an Engine API RPC class for every Engine API method used within EEST based hive
    simulators.
    """

    def new_payload(
        self,
        engine_new_payload: Union[
            EngineCancun.NewPayloadV3,
            EngineShanghai.NewPayloadV2,
            EngineParis.NewPayloadV1,
        ],
    ):
        """
        `engine_newPayloadVX`: Attempts to execute the given payload on an execution client.
        """
        return self.post_request(
            f"engine_newPayloadV{engine_new_payload.version()}", engine_new_payload.to_json_rpc()
        )

    def forkchoice_updated(
        self,
        forkchoice_state: ForkchoiceStateV1,
        payload_attributes: PayloadAttributes,
        version: int = 1,
    ):
        """
        `engine_forkchoiceUpdatedVX`: Updates the forkchoice state of the execution client.
        """
        payload_params = [forkchoice_state, payload_attributes]
        return self.post_request(f"engine_forkchoiceUpdatedV{version}", payload_params)
