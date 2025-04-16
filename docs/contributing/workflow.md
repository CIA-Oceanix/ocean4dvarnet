# Contributing Workflow

---
## **Worflow**

The Contributing Workflow used is  GithubFlow. It relies on a single main branch that is always deployable. 

For each new task (feature, fix, or improvement), a developer creates a dedicated branch (feature-name), pushes commits to it, and opens a pull request (PR) to main. After code review and approval, the branch is merged into main, and the changes can be deployed.

To avoid giving access rights, contributors work on their own forked repo, and open a pull request on the Ocean4DVarNet repository.

*More informations about githubflow workflow on [https://docs.github.com/en/get-started/using-github/github-flow](https://docs.github.com/en/get-started/using-github/github-flow)*

---
## **Contributing**

For contributing to the development of the Ocean4DVarNet packages, please follow these steps:

1. Open an issue before starting a feature or bug fix to discuss the approach with maintainers and other users.
2. Fork the Ocean4DVarNet repository on GitHub to your personal/organisation account
3. Clone your forked repository on your local machine, and configure it to stay updated with the upstream Ocean4DVarNet repository *See [Setting Up Your Fork](./fork.md)*
4. Set up the development environment. *See [Development environment set up](./dev-env-set-up.md)*
5. Create a new branch for your developments. *see [Branch Guidelines](./guidelines.md)*
6. Make your changes. Don't forget the documentation *see [Documentation](./documentation.md)*
7. Commit your change, and push your branch to your fork on GitHub. *see [Commit and PR Message Guidelines](./guidelines.md)*.
8. Open a Pull Request against the main branch of the original repository. *see [Commit and PR Message Guidelines](./guidelines.md)*
9. Request a review from maintainers or other contributors, which will follow the Code Review Process.


---
## **Code Review Process**

The Ocean4DVarNet packages have a set of automated checks to enforce coding guidelines. These checks are run via GitHub Actions on every Pull Request.

For security reasons, maintainers must review code changes before enabling automated checks.

1. Ensure that all the Development Guidelines criteria are met before submitting a Pull Request. *see [Contributing](../contributing.md)*
2. Request a review from maintainers or other contributors, noting that support is on a best-efforts basis.
3. After an initial review, a maintainer will enable automated checks to run on the Pull Request.
4. Reviewers may provide feedback or request changes to your contribution.
5. Once approved, a maintainer will merge your Pull Request into the appropriate branch.