from uuid import UUID

import pendulum
import pytest
import sqlalchemy as sa
from sqlalchemy.sql.expression import or_

from prefect.server import models, schemas
from prefect.server.database import orm_models
from prefect.server.schemas import states
from prefect.server.services.cancellation_cleanup import CancellationCleanup

NON_TERMINAL_STATE_CONSTRUCTORS = {
    states.StateType.SCHEDULED: states.Scheduled,
    states.StateType.PENDING: states.Pending,
    states.StateType.RUNNING: states.Running,
    states.StateType.PAUSED: states.Paused,
    states.StateType.CANCELLING: states.Cancelling,
}

TERMINAL_STATE_CONSTRUCTORS = {
    states.StateType.COMPLETED: states.Completed,
    states.StateType.FAILED: states.Failed,
    states.StateType.CRASHED: states.Crashed,
    states.StateType.CANCELLED: states.Cancelled,
}

THE_PAST = pendulum.now("UTC") - pendulum.Duration(hours=5)
THE_ANCIENT_PAST = pendulum.now("UTC") - pendulum.Duration(days=100)


@pytest.fixture
async def cancelled_flow_run(session, flow):
    async with session.begin():
        return await models.flow_runs.create_flow_run(
            session=session,
            flow_run=schemas.core.FlowRun(
                flow_id=flow.id, state=states.Cancelled(), end_time=THE_PAST
            ),
        )


@pytest.fixture
async def old_cancelled_flow_run(session, flow):
    async with session.begin():
        return await models.flow_runs.create_flow_run(
            session=session,
            flow_run=schemas.core.FlowRun(
                flow_id=flow.id, state=states.Cancelled(), end_time=THE_ANCIENT_PAST
            ),
        )


@pytest.fixture
async def all_states_subflow_run(session, flow, flow_run):
    async with session.begin():
        cancelling_parent_task_run = await models.task_runs.create_task_run(
            session=session,
            task_run=schemas.core.TaskRun(
                flow_run_id=flow_run.id,
                task_key="a cancelling parent task",
                dynamic_key="a cancelling parent dynamic key",
                state=states.Cancelling(),
            ),
        )
        combined_state_subflow_constructors = {
            **NON_TERMINAL_STATE_CONSTRUCTORS,
            **TERMINAL_STATE_CONSTRUCTORS,
        }
        for _, state_constructor in combined_state_subflow_constructors.items():
            await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    state=state_constructor(),
                    parent_task_run_id=cancelling_parent_task_run.id,
                ),
            )
    return cancelling_parent_task_run


@pytest.fixture
async def orphaned_task_run_maker(session):
    async def task_run_maker(flow_run, state_constructor):
        async with session.begin():
            return await models.task_runs.create_task_run(
                session=session,
                task_run=schemas.core.TaskRun(
                    flow_run_id=flow_run.id,
                    task_key="a task",
                    dynamic_key="a dynamic key",
                    state=state_constructor(),
                ),
            )

    return task_run_maker


@pytest.fixture
async def orphaned_subflow_run_maker(session, flow):
    async def subflow_run_maker(flow_run, state_constructor):
        async with session.begin():
            virtual_task = await models.task_runs.create_task_run(
                session=session,
                task_run=schemas.core.TaskRun(
                    flow_run_id=flow_run.id,
                    task_key="a virtual task",
                    dynamic_key="a virtual dynamic key",
                    state=state_constructor(),
                ),
            )

            return await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    parent_task_run_id=virtual_task.id,
                    state=state_constructor(),
                    end_time=THE_PAST,
                ),
            )

    return subflow_run_maker


@pytest.fixture
async def orphaned_subflow_run_from_deployment_maker(session, flow, deployment):
    async def subflow_run_maker(flow_run, state_constructor):
        async with session.begin():
            virtual_task = await models.task_runs.create_task_run(
                session=session,
                task_run=schemas.core.TaskRun(
                    flow_run_id=flow_run.id,
                    task_key="a virtual task for subflow from deployment",
                    dynamic_key="a virtual dynamic key for subflow from deployment",
                    state=state_constructor(),
                ),
            )

            return await models.flow_runs.create_flow_run(
                session=session,
                flow_run=schemas.core.FlowRun(
                    flow_id=flow.id,
                    parent_task_run_id=virtual_task.id,
                    state=state_constructor(),
                    end_time=THE_PAST,
                    deployment_id=deployment.id,
                    infrastructure_pid="my-pid-42",
                ),
            )

    return subflow_run_maker


async def test_all_state_types_are_tested():
    assert set(NON_TERMINAL_STATE_CONSTRUCTORS.keys()).union(
        set(TERMINAL_STATE_CONSTRUCTORS.keys())
    ) == set(states.StateType)


@pytest.mark.parametrize("state_constructor", NON_TERMINAL_STATE_CONSTRUCTORS.items())
async def test_service_cleans_up_nonterminal_runs(
    session,
    cancelled_flow_run,
    orphaned_task_run_maker,
    orphaned_subflow_run_maker,
    orphaned_subflow_run_from_deployment_maker,
    state_constructor,
):
    orphaned_task_run = await orphaned_task_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    orphaned_subflow_run = await orphaned_subflow_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    orphaned_subflow_run_from_deployment = (
        await orphaned_subflow_run_from_deployment_maker(
            cancelled_flow_run, state_constructor[1]
        )
    )
    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == state_constructor[0]
    assert orphaned_subflow_run.state.type == state_constructor[0]
    assert orphaned_subflow_run_from_deployment.state.type == state_constructor[0]

    await CancellationCleanup().start(loops=1)
    await session.refresh(orphaned_task_run)
    await session.refresh(orphaned_subflow_run)
    await session.refresh(orphaned_subflow_run_from_deployment)

    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == "CANCELLED"
    assert orphaned_subflow_run.state.type == "CANCELLED"
    assert orphaned_subflow_run_from_deployment.state.type == "CANCELLING"


@pytest.mark.parametrize("state_constructor", NON_TERMINAL_STATE_CONSTRUCTORS.items())
async def test_service_ignores_old_cancellations(
    session,
    old_cancelled_flow_run,
    orphaned_task_run_maker,
    orphaned_subflow_run_maker,
    orphaned_subflow_run_from_deployment_maker,
    state_constructor,
):
    orphaned_task_run = await orphaned_task_run_maker(
        old_cancelled_flow_run, state_constructor[1]
    )
    orphaned_subflow_run = await orphaned_subflow_run_maker(
        old_cancelled_flow_run, state_constructor[1]
    )
    orphaned_subflow_run_from_deployment = (
        await orphaned_subflow_run_from_deployment_maker(
            old_cancelled_flow_run, state_constructor[1]
        )
    )
    assert old_cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == state_constructor[0]
    assert orphaned_subflow_run.state.type == state_constructor[0]
    assert orphaned_subflow_run_from_deployment.state.type == state_constructor[0]

    await CancellationCleanup().start(loops=1)
    await session.refresh(orphaned_task_run)
    await session.refresh(orphaned_subflow_run)
    await session.refresh(orphaned_subflow_run_from_deployment)

    # tasks are ignored, but subflows will still be cancelled
    assert old_cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == state_constructor[0]
    assert orphaned_subflow_run.state.type == "CANCELLED"
    assert orphaned_subflow_run_from_deployment.state.type == "CANCELLING"


@pytest.mark.parametrize("state_constructor", TERMINAL_STATE_CONSTRUCTORS.items())
async def test_service_leaves_terminal_runs_alone(
    session,
    cancelled_flow_run,
    orphaned_task_run_maker,
    orphaned_subflow_run_maker,
    orphaned_subflow_run_from_deployment_maker,
    state_constructor,
):
    orphaned_task_run = await orphaned_task_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    orphaned_subflow_run = await orphaned_subflow_run_maker(
        cancelled_flow_run, state_constructor[1]
    )
    orphaned_subflow_run_from_deployment = (
        await orphaned_subflow_run_from_deployment_maker(
            cancelled_flow_run, state_constructor[1]
        )
    )

    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == state_constructor[0]
    assert orphaned_subflow_run.state.type == state_constructor[0]
    assert orphaned_subflow_run_from_deployment.state.type == state_constructor[0]

    await CancellationCleanup().start(loops=1)
    await session.refresh(orphaned_task_run)
    await session.refresh(orphaned_subflow_run)
    await session.refresh(orphaned_subflow_run_from_deployment)

    assert cancelled_flow_run.state.type == "CANCELLED"
    assert orphaned_task_run.state.type == state_constructor[0]
    assert orphaned_subflow_run.state.type == state_constructor[0]
    assert orphaned_subflow_run_from_deployment.state.type == state_constructor[0]


async def test_clean_up_cancelled_subflow_runs(session, all_states_subflow_run):
    cancelling_parent_task_run = all_states_subflow_run

    high_water_mark = UUID(int=0)
    fetch_subflows_query = (
        sa.select(orm_models.FlowRun)
        .where(
            or_(
                orm_models.FlowRun.state_type == states.StateType.PENDING,
                orm_models.FlowRun.state_type == states.StateType.SCHEDULED,
                orm_models.FlowRun.state_type == states.StateType.RUNNING,
                orm_models.FlowRun.state_type == states.StateType.PAUSED,
                orm_models.FlowRun.state_type == states.StateType.CANCELLING,
            ),
            orm_models.FlowRun.id > high_water_mark,
            orm_models.FlowRun.parent_task_run_id.is_not(None),
        )
        .order_by(orm_models.FlowRun.id)
        .limit(10)
    )

    async with session.begin():
        result = await session.execute(fetch_subflows_query)
        result = result.scalars().all()

    assert len(result) == 5
    assert all(
        subflow.parent_task_run_id == cancelling_parent_task_run.id
        for subflow in result
    )
