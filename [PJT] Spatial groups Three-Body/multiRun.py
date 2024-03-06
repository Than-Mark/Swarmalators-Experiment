def run_model(model):
        model.run(10)


if __name__ == "__main__":
    import numpy as np
    from itertools import product
    from main import ThreeBody
    from multiprocessing import Pool


    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.1), np.arange(1, 2.1, 1)
    ])

    savePath = "./data"

    models = [
        ThreeBody(l, d0, agentsNum=200, boundaryLength=5,
                tqdm=True, savePath=savePath, overWrite=True)
        for l, d0 in product(rangeLambdas, distanceDs)
    ]

    with Pool(4) as p:
        p.map(run_model, models)
