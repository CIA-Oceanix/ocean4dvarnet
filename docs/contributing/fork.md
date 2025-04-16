# Setting Up Your Fork

When working with a fork, follow these steps to set up your local development environment:

If you need help with git, have a look at the [Git QuickSheet](./git-quicksheet.md).


---
## **Fork the repository: Create your own copy of the repository on GitHub.**

1. On GitHub, navigate to the Ocean4DVarNet repository : [https://github.com/CIA-Oceanix/ocean4dvarnet](https://github.com/CIA-Oceanix/ocean4dvarnet)
2. On the top-right corner of the page, click `Fork`.
![](./img/fork.png)
1. Select `create a new fork`
2. Under "Owner", select your login in the dropdown menu
3. Under "Repository Name" don't use the same name as the upstream repository `ocean4dvarnet`, but customize it to distinguish it further. Keep a name including "ocean4dvarnet", ie `my-ocean4dvarnet`
4. In the "Description" field, type a description of your fork i.e "Fork of the  ocean4dvarnet package"
5. Select `Copy the DEFAULT branch only.`. you only need to copy the default branch. If you do not select this option, all branches will be copied into the new fork.
6. Then click on `Create fork`, and wait. *"It should only take a few seconds"*.


A this step, you have your own fork of the Ocean4DVarNet repository on you github account.

More informations following this [GitHub forking a repository tutorial](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository).


---
## **Clone your fork: Download your forked repository to your local machine**

1. On GitHub, navigate to your fork of the Ocean4DVarNet repository, click on `<>Code` and get the SSH url.
2. Clone your fork of the Ocean4DVarNet repository on you local machine
``` bash
$ git clone https://github.com/YOUR-USERNAME/my-ocean4dvarnet
> Cloning into 'my-ocean4dvarnet'...
> remote: Enumerating objects: 99, done.
> remote: Counting objects: 100% (99/99), done.
> remote: Compressing objects: 100% (65/65), done.
> remote: Total 99 (delta 58), reused 70 (delta 33), pack-reused 0 (from 0)
> Receiving objects: 100% (99/99), 50.73 KiB | 16.91 MiB/s, done.
> Resolving deltas: 100% (58/58), done.
```
3. To go up created directory :
```bash
$ cd my-ocean4dvarnet
```

More informations following [this section of the tutorial](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#cloning-your-forked-repository).

---
## **Keep your fork Up to date : Configuring Git to sync your fork with the upstream repository**


1. Display informations about the remote connexion of your fork repository : it is not connected to the upstream repository anymore, so it can not fetch updates.
``` bash
$ git remote -v
> origin    git@github.com:YOUR-USERNAME/my-ocean4dvarnet.git (fetch)
> origin    git@github.com:YOUR-USERNAME/my-ocean4dvarnet.git (push)
```
2. Add the upstream remote, to connect your local repository to the original repository
``` bash
$ git remote add upstream https://github.com/CIA-Oceanix/ocean4dvarnet.git
```
3. Verify the change with
``` bash
$ git remote -v
> origin	git@github.com:YOUR-USERNAME/my-ocean4dvarnet.git (fetch)
> origin	git@github.com:YOUR-USERNAME/my-ocean4dvarnet.git (push)
> upstream	https://github.com/CIA-Oceanix/ocean4dvarnet.git (fetch)
> upstream	https://github.com/CIA-Oceanix/ocean4dvarnet.git (push)
```
4. Prevent accidental pushes to upstream: After setting up your fork and configuring the original repository as an upstream remote, it’s a good practice to prevent accidental pushes to the upstream repository. You can do this by explicitly setting the push URL of the upstream remote to no_push.
``` bash
$ git remote set-url --push upstream no_push
```
5. Verify the change with
``` bash
$ git remote -v
> origin      git@github.com:YOUR-USERNAME/my-ocean4dvarnet.git (fetch)
> origin      git@github.com:YOUR-USERNAME/my-ocean4dvarnet.git (push)
> upstream    https://github.com/CIA-Oceanix/ocean4dvarnet.git (fetch)
> upstream    no_push (push)
```

Now, you can keep your fork synced with the upstream repository with 
``` bash
$ git fetch upstream main
> From https://github.com/CIA-Oceanix/ocean4dvarnet
> * branch            main       -> FETCH_HEAD
```

With this configuration, you can still fetch updates from the upstream repository but won’t be able to accidentally push changes to it.

More informations by following [this section of the tutorial](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#configuring-git-to-sync-your-fork-with-the-upstream-repository).

