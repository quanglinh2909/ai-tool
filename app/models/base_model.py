from datetime import datetime, timezone

from piccolo.table import Table
from piccolo.columns import UUID, Timestamptz
import uuid


class BaseModel(Table):
    id = UUID(primary_key=True, default=uuid.uuid4)
    created_at = Timestamptz(default=lambda: datetime.now(timezone.utc))
    updated_at = Timestamptz(default=lambda: datetime.now(timezone.utc),
                             on_update=lambda: datetime.now(timezone.utc))

