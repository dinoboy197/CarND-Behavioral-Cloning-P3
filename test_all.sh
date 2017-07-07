#!/usr/bin/env bash

git reset --hard HEAD > /dev/null
git checkout -q master > /dev/null
shas=()

array_not_contains () {
    local seeking=$1; shift
    local in=0
    for element; do
        if [[ $element == $seeking ]]; then
            in=1
            break
        fi
    done
    return $in
}

# only look at master
branches=('master')
#eval "$(git for-each-ref --shell --format='branches+=(%(refname:short))' refs/heads/)"
for branch in "${branches[@]}"; do
    # loop through all commits by Taylor
    git checkout -q $branch > /dev/null
    for sha in $(git rev-list HEAD --author='Taylor Raack' --abbrev-commit); do
        if array_not_contains $sha "${shas[@]}"; then
            shas+=($sha)
            # loop through training data
            for tdata in 'origdata' 'tdata1'; do
                identifier="$sha-$tdata"
                model_file="model-$identifier.h5"
                stats_file="stats-$identifier"
        
                # if model or stats file are not found
                if [ ! -f $model_file ] || [ ! -f $stats_file ]; then
                    echo "computing $identifier"
                    git reset --hard HEAD > /dev/null
                    git checkout -q $sha > /dev/null
                    rm -f data
                    ln -s $tdata data
                    touch $model_file
                    python model.py > $stats_file 2>&1
                    mv model.h5 $model_file
                fi
            done
        fi
    done
done
