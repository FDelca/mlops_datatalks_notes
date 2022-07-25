# Setting deployment specifications
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner # With this we will run it without a container

DeploymentSpec(
    flow_location="score.py",
    name='ride_duration_prediction',
    parameters={
        "taxy_type":"green",
        "run_id":"0b929405a77e4097ab00f5db2a117412"
    },
    flow_storage='24e09b7f-8a05-465b-ba55-74c858feba78', # configure prefect storage - `prefect storage create`
    schedule=CronSchedule(cron='0 3 2 * *'), # At 03:00 on day-of-month 2.
    flow_runner=SubprocessFlowRunner(),
    tags=['ml-in-action'],
)


# queue: 8c91e720-9667-4154-bdfe-48c4c67d9c19
# start an agent:
## prefect agent start 8c91e720-9667-4154-bdfe-48c4c67d9c19