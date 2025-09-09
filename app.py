from datetime import datetime

from aws_cdk import (
    App,
    CfnOutput,
    Duration,
    Environment,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from constructs import Construct

project_name = "dissertation-ui-project"
cur_dt = datetime.now().strftime("%H:%M:%S %d-%m-%Y")


class PersonaInterviewServiceStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        _handler = lambda_.Function(
            self,
            f"{project_name}-handler",
            function_name=f"{project_name}-handler",
            description=f"Deployed at {cur_dt}",
            memory_size=128,
            timeout=Duration.minutes(5),
            handler="main.handler",
            code=lambda_.Code.from_asset("app"),
            runtime=lambda_.Runtime.PYTHON_3_12,
            allow_public_subnet=True,
        )

        # Adding due to APIGW timeout when using gemini
        _handler.add_function_url(auth_type=lambda_.FunctionUrlAuthType.NONE)

        # Add IAM permissions to the Lambda function
        _handler.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "lambda:*",
                ],
                resources=["*"],
            )
        )

        # Outputs for the stack
        CfnOutput(self, f"{project_name}-lambda-name", value=_handler.function_name)


app = App()
PersonaInterviewServiceStack(
    app,
    f"{project_name}-stack",
    env=Environment(account="878185805840", region="eu-west-2"),
    description=f"MSc dissertation project deployed at {cur_dt}",
)
app.synth()
