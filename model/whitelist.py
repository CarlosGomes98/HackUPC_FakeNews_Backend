import itertools
trusted_sources = ["theguardian", "bbc", "apnews", "edition.cnn"]

domains = [".com", ".co.uk", ".es",]

def is_from_trusted_source(url):
    print([a+b for a, b in itertools.product(trusted_sources,domains)])
    return url in [a+b for a, b in itertools.product(trusted_sources,domains)]