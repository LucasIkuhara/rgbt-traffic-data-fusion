build-pub:
    docker build -t tex-env -f publication/pandoc.Dockerfile publication

pub template="springer": build-pub
    @python3 publication/concat.py && \
    docker run --rm \
       --volume "$(pwd)/publication:/data" \
       --user $(id -u):$(id -g) \
       tex-env tmp.md -o out.pdf \
       --template {{ template }} \
       --syntax-highlighting idiomatic

dataset:
    curl -L -o aau-rainsnow.zip \
    https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow

run: dataset
    poetry run python -m src.predict
    poetry run python -m src.evaluate >> results.txt

visualize:
    poetry run python -m src.visualize