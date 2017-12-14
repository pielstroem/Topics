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

## Committing Jupyter Notebooks

Our Jupyter Notebooks should be committed without their output cells and execution counts since generated output of
probalistic methods makes version control unneccessarily complicated. The script <ipynb_drop_output.py> makes output
stripping easy, but requires some preparation:

### Preparation for each contributor machine

Every contributor should:

1. place the <ipynb_drop_output.py> script somewhere on the `$PATH`
2. make it executable (`chmod a+x ipynb_drop_output.py`)
3. configure a git filter called `clean_ipynb` as follows:

```bash
git config --global filter.clean_ipynb.clean ipynb_drop_output.py
git config --global filter.clean_ipynb.smudge cat
```

This will edit your `~/.gitconfig` so you globally have this filter available. The filter is linked to notebooks using
an entry in the `.gitattributes` file, this is already configured for this project.

### Preparation of each notebook

Each notebook needs a metadata entry to enable the filter. Open the notebook in Jupyter, click _Edit / Edit Notebook Metadata_
and add the following entry to the outermost dictionary:

```json
  "git": {
    "suppress_outputs": true
  },
```

Now, when you add this notebook to the index or try to run a diff on it, the script is used to prepare a clean version.
Your working copy will not be modified, so _you_ will see the output cells and execution counts you generated.

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
