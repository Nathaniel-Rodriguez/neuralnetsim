export SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance,special-members"
sphinx-apidoc -o . ../src/neuralnetsim --separate --implicit-namespaces --module-first -f