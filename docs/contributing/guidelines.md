# Guidelines

---
## **Branching Guidelines**

- Use feature branches for new features (e.g., feat/your-feature)
- Use fix branches for bug fixes (e.g., fix/your-bug)
- Use a descriptive name that indicates the purpose of the branch
- Keep branches up to date with main before opening a Pull Request

---
## **Commit and PR Message Guidelines**

PR and commit Message must follows the [Conventional Commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/), because our automated release process ([release-please](https://github.com/googleapis/release-please)) relies on conventional commits to generate changelogs and determine version bumps automatically.

Please ensure that it also follows theses guidelines :

1. Reference relevant issue numbers in commit messages when applicable (e.g., “fix: resolve data loading issue #123”).
3. Use present tense and imperative mood in commit messages (e.g., `Add feature` not `Added feature`).


The PR title will become the squash commit message when merged to main

For commits, we furthermore encourage you to make small, focused commits with clear and concise messages.

The format of [Conventional Commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/). is: `type[(scope)][!]: description`. 

 `!` after the type/scope to indicate a breaking change.

Common types include:

- `feat`: New feature.
- `fix`: Bug fix.
- `docs`: Documentation only.
- `style`: Code style/format changes.
- `refactor`: Code changes that neither fix bugs nor add features.
- `test`: Adding or modifying tests.
- `perf`: About performance.
- `deps`: Dependencies update.
- `chore`: Maintenance tasks (ackaging or repo/files management.)
- `build`: About packaging, building wheels, etc.
- `ci`: About Continuous Integration.

For example:

- `feat(train): add new loss function`
- `fix(plot): resolve node indexing bug`
- `docs(readme): update installation steps`
- `feat(models)!: change model input format (breaking change)`
- `refactor!: restructure project layout (breaking change)`
