from uuid import UUID

from pydantic import ConfigDict

from prefect._internal.schemas.bases import PrefectBaseModel
from prefect.client.schemas.objects import FlowRun


class WorkerFlowRunResponse(PrefectBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    work_pool_id: UUID
    work_queue_id: UUID
    flow_run: FlowRun