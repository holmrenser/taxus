Deploying to pypi:

```
# check current version
poetry version

# update version in taxus/__init__.py

# tests and coverage reports
poetry run coverage run
poetry run coverage report

# build
poetry build

# publish
poetry publish
```
