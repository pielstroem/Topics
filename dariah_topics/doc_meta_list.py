from pathlib import Path
from abc import ABC, abstractmethod
import glob
import os
import pandas as pd
import regex
from copy import deepcopy

class BaseDocList(ABC):

    def __init__(self, basepath, default_pattern='{Index}.txt'):
        self.basepath = Path(basepath)
        self.default_pattern = default_pattern
        self._segment_counts = None

    @abstractmethod
    def get_docs(self): #eq. get_docs, list_docs
        pass

    def copy(self):
        return deepcopy(self)

    def get_full_path(self, document, as_str=False): #eq. full_path
        path = Path(self.basepath, document)
        if as_str:
            path = str(path)
        return path

    def get_full_paths(self, as_str=False):
        return [self.get_full_path(doc, as_str) for doc in self.get_docs()]

    def flatten_segments(self, segmented_docs): # flatten_segments*2
        segment_counts = []
        self._segment_counts = segment_counts
        for doc in segmented_docs:
            segment_counts.append(0)
            for segment in doc:
                segment_counts[-1] += 1
                yield segment

    def yield_segments(self): # segments, list_segments
        if self.segment_counts() is None:
            raise ValueError("Not segmentized yet")
        for doc, segment_count in self.get_docs(), self.segment_counts():
            for segment_no in range(segment_count):
                yield dict(doc, segment_no)

    def segment_counts(self): #doclist only
        return self._segment_counts

    #filenames, yields filenames
    def yield_filenames(self, basepath=None, pattern=None, segments=False):
        if basepath is None:
            basepath = self.basepath
        if pattern is None:
            pattern = self.default_pattern
        items = self.yield_segments() if segments else self.get_docs()
        for doc in items:
            filename = os.path.join(basepath, pattern.format_map(doc))
            yield filename

    def forall(self, function, *args, basedir=None, pattern=None, segments=False, **kwargs):
        for args in zip(self.yield_filenames(basedir, pattern, segments), *args):
            yield function(*args, **kwargs)

    def segment_filenames(self,
                          format="{path.stem}.{segment:0{maxwidth}d}{path.suffix}",
                          basepath=None,
                          as_str=False): # segment_filenames (doc), yields path for segments
        segment_counts = self.segment_counts()
        if segment_counts is None:
            raise ValueError("No segments recorded.")
        maxwidth = len(str(max(segment_counts)))
        if basepath is None:
            basepath = self.basepath

        for document, segment_no in self.segment_counts():
            filename = format.format(path=document, maxwidth=maxwidth,
                                     segment=segment_no)
            segment_path = Path(basepath, filename)
            if as_str:
                yield str(segment_path)
            else:
                yield segment_path

    def __iter__(self):
        return iter(self.get_full_paths(as_str=True))

    def __len__(self):
        return len(self.get_docs())

    def __getitem__(self, index):
        try:
            selection = self.__getitem__(index)
        except AttributeError:
            selection = self.get_docs()[index]

        if isinstance(index, slice):
            return [self.get_full_path(doc, as_str=True) for doc in selection]
        else:
            return self.get_full_path(selection, as_str=True)


class DocCorpus(BaseDocList):

    def __init__(self, basepath, glob_pattern='*', filenames=None, **kwargs):
        super().__init__(**kwargs)
        self.basepath = Path(basepath)
        self._segment_counts = None
        if filenames is None:
            self._files = [p.relative_to(self.basepath)
                           for p in self.basepath.glob(glob_pattern)]
        else:
            paths = (Path(name) for name in filenames)
            if glob_pattern is not None:
                paths = (path for path in paths if path.match(glob_pattern))
            self._files = list(paths)

    def get_docs(self):
        return self._files

    def with_segment_files(self, basepath=None, **kwargs):
        segment_counts = self.segment_counts()
        if segment_counts is None:
            raise ValueError("No segments recorded.")
        if basepath is None:
            basepath = self.basepath
        result = self.copy()
        result._segment_counts = 0
        result.basepath = basepath
        result._files = list(self.segment_filenames(basepath='', **kwargs))
        return result


class MetaCorpus(BaseDocList):

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.metadata = pd.DataFrame(data)

    def get_docs(self):
        return (t._asdict() for t in self.metadata.itertuples())

def fn2metadata(glob_pattern='corpus/*.txt', fn_pattern=regex.compile('(?<author>[^_]+)_(?<title>.+)'), index=None):
    metadata_list = []
    for filename in glob.glob(glob_pattern):
        basename, __ = os.path.splitext(os.path.basename(filename))
        md = fn_pattern.match(basename).groupdict()
        md["basename"] = basename
        md["filename"] = filename
        metadata_list.append(md)
    metadata = pd.DataFrame(metadata_list)
    if index is not None:
        metadata = metadata.set_index(index)
    return metadata