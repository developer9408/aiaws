#!/usr/bin/env sh

if [ "$TRAVIS_BRANCH" = "master" ]
then
    echo "prod"
elif [ "$TRAVIS_BRANCH" = "staging" ]
then
    echo "staging"
else
    echo "dev"
fi
