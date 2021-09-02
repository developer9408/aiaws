#!/usr/bin/env sh

TASK_DEFINITION=$(aws ecs describe-task-definition --region eu-central-1 --task-definition "${TASK_DEFINITION_NAME}")
NEW_TASK_DEFINITION=$(echo ${TASK_DEFINITION} | python ${TRAVIS_BUILD_DIR}/deploy/update_ecs_task_definition.py ${DOCKER_REPO_URL}:${CONTAINER_TAG})
EXEC_ROLE_ARN=$(echo ${NEW_TASK_DEFINITION} | jq -r .executionRoleArn)
NEW_CONTAINER_DEFINITION=$(echo ${NEW_TASK_DEFINITION} | jq -r .containerDefinitions)
REGISTER_RESULT=$(aws ecs register-task-definition --family "${TASK_DEFINITION_NAME}" --container-definitions "${NEW_CONTAINER_DEFINITION}" --execution-role-arn "${EXEC_ROLE_ARN}")
echo ${REGISTER_RESULT}
REVISION=$(echo ${REGISTER_RESULT} | jq -r .taskDefinition.revision)
aws ecs update-service --cluster "${CLUSTER_NAME}" --service "${SERVICE_NAME}" --task-definition "${TASK_DEFINITION_NAME}:${REVISION}"

