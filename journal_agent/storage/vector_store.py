from datetime import datetime

from journal_agent.model.session import Fragment, Tag


def fragment_to_chroma(f: Fragment) -> dict:
    return {
        "id": f.id,
        "document": f.content,
        "metadata": {
            "session_id": f.session_id,
            "exchange_ids": ",".join(f.exchange_ids),
            "tags": ",".join(t.tag for t in f.tags),
            "timestamp": f.timestamp.isoformat(),
        },
    }

def fragment_from_chroma(row: dict) -> Fragment:
    meta = row["metadata"]
    return Fragment(
        id=row["id"],
        content=row["document"],
        session_id=meta["session_id"],
        exchange_ids=meta["exchange_ids"].split(",") if meta["exchange_ids"] else [],
        tags=[Tag(tag=t) for t in meta["tags"].split(",")] if meta["tags"] else [],
        timestamp=datetime.fromisoformat(meta["timestamp"]),
    )