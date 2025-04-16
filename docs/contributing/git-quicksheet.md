# Git QuickSheet

- Clone a repository
``` bash
$ git clone https://github.com/YOUR-USERNAME/my-ocean4dvarnet
```
- Display informations about the remote connexion of a repository.
``` bash
$ git remote -v
```
- Display local repository state informations 
``` bash
$ git status
```
- List existing branches
``` bash
git branch -l
```
- Select a branch
``` bash
git switch BRANCH_NAME
```
- Create a branch
``` bash
git branch BRANCH_NAME
```
- Delete a branch
``` bash
git branch -d BRANCH_NAME
```
- Sync `main` branch of the local repository with the remote upstream repository
    - Only the index 
    ```bash
    git fetch upstream main
    ```
    - Index and workspace (equiv to `git fetch`, and `git merge`)
    ```bash
    git pull upstream main
    ```
- Update a branch with the content of the main branch. 
    - Solution 1:  Re-apply the commit of the main branch to the BRANCH_NAME branch. It rewrite history, create new commits. This is the preferred solution, but the modification of commits ID can be a problem when collaborating on a branch.
    ``` bash
    git switch BRANCH_NAME
    git merge main
    ```
    - Solution 2 : Create a new commit with the result of the merge. It Preserve branch history.
    ``` bash
    git switch BRANCH_NAME
    git merge main
    ```
