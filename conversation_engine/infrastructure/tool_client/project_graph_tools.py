"""
project_spec tool — CRUD operations on ProjectSpecification for the ReAct agent.

The LLM interacts with ``ProjectSpecification`` (flat, name-based references).
It never sees nodes, edges, or IDs.  The tool validates references via the
snapshot facade and persists specs through the ``ProjectStore``.

Methods:
    CREATE(payload)      — persist a new project from a ProjectSpecification
    READ(key)            — retrieve an existing project as a ProjectSpecification
    UPDATE(key, payload) — full-replace the spec, preserving control fields
    DELETE(key)          — remove a project by name
"""

from __future__ import annotations

from enum import StrEnum

from commons.tool_client import ToolSpec
from pydantic import BaseModel, Field

from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.storage import ProjectStore
from conversation_engine.storage.snapshot import ProjectSnapshot
from conversation_engine.storage.snapshot_facade import (
    SnapshotConversionError,
    snapshot_to_graph,
)


# ── I/O models ─────────────────────────────────────────────────────
class CRUDMethod(StrEnum):
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class ProjectGraphInput(BaseModel):
    """Input for the project_spec tool."""

    method: CRUDMethod = Field(
        ...,
        description=(
            "The operation to perform: "
            "CREATE to persist a new project, "
            "READ to retrieve an existing project, "
            "UPDATE to full-replace the spec (preserves rules/quiz/prompt), "
            "DELETE to remove a project."
        ),
    )
    key: str | None = Field(
        None,
        description=(
            "Project name — required for READ, UPDATE, and DELETE. "
            "For CREATE, the project name is taken from the payload."
        ),
    )
    payload: ProjectSnapshot | None = Field(
        None,
        description=(
            "The project data — required for CREATE and UPDATE. Omit for READ and DELETE."
        ),
    )


class ProjectGraphOutput(BaseModel):
    """Output from the project_spec tool."""

    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(default="", description="Human-readable status message")
    payload: ProjectSnapshot | None = Field(
        None,
        description="The project data — returned by READ and UPDATE, None for CREATE and DELETE.",
    )


# ── CRUD operations ───────────────────────────────────────────────────


def _create_project(
    project_store: ProjectStore,
    snapshot: ProjectSnapshot,
) -> ProjectGraphOutput:
    """Create a new project from a snapshot."""
    if project_store.exists(snapshot.project_name):
        return ProjectGraphOutput(
            success=False,
            message=f"Project '{snapshot.project_name}' already exists. "
            "DELETE it first, then CREATE again.",
        )
    try:
        # Validate refs by attempting a graph conversion (but don't store the graph)
        snapshot_to_graph(snapshot)
    except SnapshotConversionError as e:
        return ProjectGraphOutput(
            success=False,
            message=f"Invalid snapshot: {e}",
        )
    config = DomainConfig(
        project_name=snapshot.project_name,
        project_spec=snapshot,
    )
    project_store.save(config)
    return ProjectGraphOutput(
        success=True,
        message=f"Project '{snapshot.project_name}' created.",
    )


def _update_project(
    project_store: ProjectStore,
    key: str,
    snapshot: ProjectSnapshot,
) -> ProjectGraphOutput:
    """Full-replace the project spec, preserving control fields (rules, quiz, prompt, etc.)."""
    existing = project_store.load(key)
    if existing is None:
        return ProjectGraphOutput(
            success=False,
            message=f"Project '{key}' not found. Use CREATE for new projects.",
        )
    try:
        snapshot_to_graph(snapshot)
    except SnapshotConversionError as e:
        return ProjectGraphOutput(
            success=False,
            message=f"Invalid snapshot: {e}",
        )
    updated = DomainConfig(
        project_name=existing.project_name,
        project_spec=snapshot,
        rules=existing.rules,
        quiz=existing.quiz,
        # query_patterns=existing.query_patterns,
        system_prompt=existing.system_prompt,
        metadata=existing.metadata,
    )
    project_store.save(updated)
    return ProjectGraphOutput(
        success=True,
        message=f"Project '{key}' updated.",
        payload=snapshot,
    )


def _read_project(
    project_store: ProjectStore,
    key: str,
) -> ProjectGraphOutput:
    """Read an existing project as a snapshot."""
    config = project_store.load(key)
    if config is None:
        return ProjectGraphOutput(
            success=False,
            message=f"Project '{key}' not found.",
        )
    if config.project_spec is None:
        return ProjectGraphOutput(
            success=True,
            message=f"Project '{key}' exists but has no project specification.",
            payload=ProjectSnapshot(project_name=key),
        )
    return ProjectGraphOutput(
        success=True,
        message=f"Project '{key}' loaded.",
        payload=config.project_spec,
    )


def _delete_project(
    project_store: ProjectStore,
    key: str,
) -> ProjectGraphOutput:
    """Delete a project by name."""
    removed = project_store.delete(key)
    if removed:
        return ProjectGraphOutput(
            success=True,
            message=f"Project '{key}' deleted.",
        )
    return ProjectGraphOutput(
        success=False,
        message=f"Project '{key}' not found.",
    )


# ── Tool factory ───────────────────────────────────────────────────


def make_project_spec_tool(project_store: ProjectStore) -> ToolSpec:
    """
    Create a project_spec ToolSpec bound to a ProjectStore.

    Parameters
    ----------
    project_store : ProjectStore
        The persistence layer to use (InMemoryProjectStore,
        FileProjectStore, future MemgraphProjectStore, etc.)
    """

    def handler(input_: ProjectGraphInput) -> ProjectGraphOutput:
        match input_.method:
            case CRUDMethod.CREATE:
                if input_.payload is None:
                    return ProjectGraphOutput(
                        success=False,
                        message="CREATE requires a payload (ProjectSpecification).",
                    )
                return _create_project(project_store, input_.payload)

            case CRUDMethod.UPDATE:
                if not input_.key:
                    return ProjectGraphOutput(
                        success=False,
                        message="UPDATE requires a key (project name).",
                    )
                if input_.payload is None:
                    return ProjectGraphOutput(
                        success=False,
                        message="UPDATE requires a payload (ProjectSpecification).",
                    )
                return _update_project(project_store, input_.key, input_.payload)

            case CRUDMethod.READ:
                if not input_.key:
                    return ProjectGraphOutput(
                        success=False,
                        message="READ requires a key (project name).",
                    )
                return _read_project(project_store, input_.key)

            case CRUDMethod.DELETE:
                if not input_.key:
                    return ProjectGraphOutput(
                        success=False,
                        message="DELETE requires a key (project name).",
                    )
                return _delete_project(project_store, input_.key)

            case _:
                return ProjectGraphOutput(
                    success=False,
                    message=f"Unknown method '{input_.method}'. Use CREATE, READ, UPDATE, or DELETE.",
                )

    return ToolSpec(
        name="project_spec",
        description=(
            "CRUD operations on a project's specification. "
            "This is the sole interface for persisting and retrieving project data.\n\n"
            "Methods:\n"
            "  CREATE(payload)      — persist a new project from a ProjectSpecification\n"
            "  READ(key)            — retrieve an existing project as a ProjectSpecification\n"
            "  UPDATE(key, payload) — full-replace the spec (rules/quiz/prompt preserved)\n"
            "  DELETE(key)          — remove a project by name\n\n"
            "The payload is a ProjectSpecification with goals, requirements, "
            "steps, constraints, and dependencies. Use name-based references "
            "(e.g. a requirement's goal_ref is the goal's name, not an ID)."
        ),
        input_model=ProjectGraphInput,
        output_model=ProjectGraphOutput,
        handler=handler,
    )


def make_project_graph_tool(project_store: ProjectStore) -> ToolSpec:
    """Legacy alias for make_project_spec_tool."""
    return make_project_spec_tool(project_store)
