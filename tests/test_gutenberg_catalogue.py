# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Selector logic, tested against fixture rows rather than the network."""
from unittest.mock import patch
import pytest
from train.corpus import CorpusSource
from scripts.build_gutenberg_catalogue import matches_source, iter_metadata

FABRE = {"authors": "Fabre, Jean-Henri", "title": "The Life of the Spider",
         "bookshelves": "Science", "subjects": "Spiders"}
STEIN = {"authors": "Stein, Gertrude", "title": "Tender Buttons",
         "bookshelves": "", "subjects": "Prose poetry"}
KIDLIT = {"authors": "Nesbit, E. (Edith)", "title": "Five Children and It",
          "bookshelves": "Children's Literature", "subjects": "Fantasy"}
UNRELATED = {"authors": "Smith, John", "title": "A Treatise on Drainage",
             "bookshelves": "Engineering", "subjects": "Sewerage"}


def _src(**kw):
    base = dict(name="t", slice="spine", target_share=0.1,
                hf_repo="r", hf_revision="rev")
    base.update(kw)
    return CorpusSource(**base)


def test_author_match_is_case_insensitive_substring():
    src = _src(authors=["fabre, jean-henri"])
    assert matches_source(FABRE, src)
    assert not matches_source(UNRELATED, src)


def test_surname_only_author_entry_matches():
    """'Dunsany' must match 'Dunsany, Lord', which is how Gutenberg records him."""
    src = _src(authors=["Dunsany"])
    assert matches_source({"authors": "Dunsany, Lord", "title": "", "bookshelves": "",
                           "subjects": ""}, src)


def test_bookshelf_match_is_case_insensitive_substring():
    src = _src(bookshelves=["children's literature"])
    assert matches_source(KIDLIT, src)
    assert not matches_source(UNRELATED, src)


def test_author_or_bookshelf_matches_when_both_given():
    """A source listing both selects the union, not the intersection."""
    src = _src(authors=["Stein, Gertrude"], bookshelves=["Children's Literature"])
    assert matches_source(STEIN, src)
    assert matches_source(KIDLIT, src)
    assert not matches_source(UNRELATED, src)


def test_no_selectors_matches_nothing_here():
    """A Gutenberg source with no selectors would pull all 48k books by accident."""
    src = _src()
    assert not matches_source(FABRE, src)


def test_missing_metadata_fields_do_not_raise():
    src = _src(authors=["Fabre"])
    assert not matches_source({}, src)
    assert not matches_source({"authors": None}, src)


def test_iter_metadata_fails_loudly_when_no_shards_found():
    """If dataset layout changes and no matching shards exist, fail explicitly."""
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_instance = mock_api.return_value
        # Return a file list with no matching parquet shards.
        mock_instance.list_repo_files.return_value = [
            "README.md", "data/metadata.json", "some_other_file.txt"
        ]
        with pytest.raises(RuntimeError, match="No parquet shards found"):
            # Attempt to iterate — should fail immediately on shard discovery.
            list(iter_metadata("some_revision"))
