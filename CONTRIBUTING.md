# Information for Contributors

## Development Environment

* [Code and Issues are at Github](http://github.com/DARIAH-DE/Topics)
* [Integration Build](http://dev.digital-humanities.de/ci/jobs/DARIAH-Topics)

### Start hacking

```bash
git clone -b testing git@github.com:DARIAH-DE/Topics
cd Topics
mkvirtualenv Topics      # if you use virtualenvwrapper
workon Topics            # if you use virtualenvwrapper
pip install -r requirement-dev.txt
```

## Releasing / Pushing to Master

The _testing_ branch is the integration branch for current developments. The _master_ branch should always contain the latest stable version. 

Pushing to master is protected, you can only push heads that have an "green" status from the integration build. To do so, do the following (from a clean working copy):

1. Prepare everything in `testing`. Don't forget to tune the version number.
2. Merge testing into master. Use `--no-ff` to make sure we get a merge commit: `git checkout master; git merge --no-ff testing`
3. if there are conflicts, resolve them and commit (to master)
4. now, fast-forward-merge master into testing: `git checkout testing; git merge master`. testing and master should now point to the same commit.
5. push testing. This will trigger the integration build, which will hopefully work out fine.
6. when the build has successfully finished, push master.

If something goes wrong, `git reset --hard master origin/master` and try again.

## Documentation

The documentation is built using [Sphinx](http://www.sphinx-doc.org/). 
The following files influence the docs:

* ``index.rst`` contains the landing page with the table of contents. Here, all files should be linked.
* ``*.ipynb`` for tutorials etc. can be linked from the index file
* ``README.md`` and ``CONTRIBUTING.md`` will also be included
* Docstrings in the modules will be included
* ``docs/**/*`` may contain additional files
* ``conf.py`` contains sphinx configuration
* ``setup.py`` contains version numbers etc.

### Documentation formats

Standalone documentation files can be written in one of the following formats:

* ReSTructured Text (`*.rst`), [see Sphinx docs](http://www.sphinx-doc.org/en/stable/rest.html)
* Jupyter Notebook (`*.ipynb`), by way of [nbsphinx](https://nbsphinx.readthedocs.io/)
* Markdown

Docstrings should follow [Google conventions](http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments), this is supported by [Napoleon](http://www.sphinx-doc.org/en/stable/ext/napoleon.html).

### Build the documentation

Run `python setup.py build_sphinx -a`, which will create the documentation tree in `build/sphinx/html`.

### After adding another module

Run `sphinx-apidoc -M -e -o docs/gen dariah_topics` to add a documentation stub to `docs/gen`. 
