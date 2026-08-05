build-pub:
    docker build -t tex-env -f publication/pandoc.Dockerfile publication

pub template="eisvogel": build-pub
    @python3 publication/concat.py && \
    docker run --rm \
       --volume "$(pwd)/publication:/data" \
       --user $(id -u):$(id -g) \
       tex-env tmp.md -o out.pdf \
       --template {{ template }} \
       --syntax-highlighting idiomatic
