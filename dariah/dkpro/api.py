from . import processing

def process(path, jar, language, reader="plaintext", xms="4g"):
    output = Path(tempfile.gettempdir(), "dariah")
    d = processing.DKPro(jar=jar,
                         xms=xms)
    d.process(input=path,
              output=output,
              language=language,
              reader=reader)
    for file in processing.DKPro.read(output):
        yield file

def filter(documents):
    for document in documents:
        pass

def corpus(do)
