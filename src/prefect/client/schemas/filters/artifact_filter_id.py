from typing import List, Optional
from uuid import UUID

from pydantic import Field

from prefect._internal.schemas.bases import PrefectBaseModel


class ArtifactFilterId(PrefectBaseModel):
    """Filter by `Artifact.id`."""

    any_: Optional[List[UUID]] = Field(
        default=None, description="A list of artifact ids to include"
    )