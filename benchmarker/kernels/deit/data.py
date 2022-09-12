from benchmarker.data.synthetic.images import ImageGen


# TODO: rewrite whole thing to just return generator
def get_data(params):
    gen = ImageGen(params)
    return gen()
