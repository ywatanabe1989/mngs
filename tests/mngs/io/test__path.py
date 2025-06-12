#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-31"
# File: test__path.py

"""Tests for mngs.io._path module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestFindTheGitRootDir:
    """Test git root directory finding functionality."""

    def test_find_git_root_in_repo(self):
        """Test finding git root in a git repository."""
        from mngs.io._path import find_the_git_root_dir

        # Mock git.Repo
        with patch("git.Repo") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.working_tree_dir = "/path/to/git/root"
            mock_repo_class.return_value = mock_repo

            result = find_the_git_root_dir()

            assert result == "/path/to/git/root"
            mock_repo_class.assert_called_once_with(".", search_parent_directories=True)

    def test_find_git_root_not_in_repo(self):
        """Test behavior when not in a git repository."""
        from mngs.io._path import find_the_git_root_dir

        # Mock git.Repo to raise exception
        with patch("git.Repo") as mock_repo_class:
            import git

            mock_repo_class.side_effect = git.exc.InvalidGitRepositoryError(
                "Not a git repo"
            )

            with pytest.raises(git.exc.InvalidGitRepositoryError):
                find_the_git_root_dir()


class TestSplitFpath:
    """Test file path splitting functionality."""

    def test_split_fpath_basic(self):
        """Test basic file path splitting."""
        from mngs.io._path import split_fpath

        fpath = "/home/user/data/file.txt"
        dirname, fname, ext = split_fpath(fpath)

        assert dirname == "/home/user/data/"
        assert fname == "file"
        assert ext == ".txt"

    def test_split_fpath_complex_extension(self):
        """Test splitting with complex extensions."""
        from mngs.io._path import split_fpath

        # Double extension (only last one is considered extension)
        fpath = "/path/to/archive.tar.gz"
        dirname, fname, ext = split_fpath(fpath)

        assert dirname == "/path/to/"
        assert fname == "archive.tar"
        assert ext == ".gz"

    def test_split_fpath_no_extension(self):
        """Test splitting file with no extension."""
        from mngs.io._path import split_fpath

        fpath = "/path/to/README"
        dirname, fname, ext = split_fpath(fpath)

        assert dirname == "/path/to/"
        assert fname == "README"
        assert ext == ""

    def test_split_fpath_root_file(self):
        """Test splitting file in root directory."""
        from mngs.io._path import split_fpath

        fpath = "/file.txt"
        dirname, fname, ext = split_fpath(fpath)

        assert dirname == "/"
        assert fname == "file"
        assert ext == ".txt"

    def test_split_fpath_relative_path(self):
        """Test splitting relative path."""
        from mngs.io._path import split_fpath

        fpath = "../data/01/day1/split_octave/2kHz_mat/tt8-2.mat"
        dirname, fname, ext = split_fpath(fpath)

        assert dirname == "../data/01/day1/split_octave/2kHz_mat/"
        assert fname == "tt8-2"
        assert ext == ".mat"

    def test_split_fpath_hidden_file(self):
        """Test splitting hidden file."""
        from mngs.io._path import split_fpath

        fpath = "/home/user/.config"
        dirname, fname, ext = split_fpath(fpath)

        assert dirname == "/home/user/"
        assert fname == ".config"
        assert ext == ""


class TestTouch:
    """Test file touch functionality."""

    def test_touch_creates_new_file(self, tmp_path):
        """Test that touch creates a new file."""
        from mngs.io._path import touch

        test_file = tmp_path / "new_file.txt"
        assert not test_file.exists()

        touch(str(test_file))

        assert test_file.exists()
        assert test_file.is_file()

    def test_touch_updates_existing_file(self, tmp_path):
        """Test that touch updates modification time of existing file."""
        from mngs.io._path import touch
        import time

        test_file = tmp_path / "existing.txt"
        test_file.write_text("content")

        # Get initial modification time
        initial_mtime = os.path.getmtime(test_file)

        # Wait a bit to ensure time difference
        time.sleep(0.1)

        # Touch the file
        touch(str(test_file))

        # Check modification time updated
        new_mtime = os.path.getmtime(test_file)
        assert new_mtime > initial_mtime

        # Content should remain unchanged
        assert test_file.read_text() == "content"

    def test_touch_nested_directory(self, tmp_path):
        """Test touch with nested directory path."""
        from mngs.io._path import touch

        nested_dir = tmp_path / "level1" / "level2"
        nested_dir.mkdir(parents=True)

        test_file = nested_dir / "file.txt"
        touch(str(test_file))

        assert test_file.exists()


class TestFind:
    """Test find functionality."""

    def test_find_files_only(self, tmp_path):
        """Test finding files only."""
        from mngs.io._path import find

        # Create test structure
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").touch()

        # Find all files
        results = find(str(tmp_path), type="f", exp="*.txt")

        assert len(results) == 3
        assert all(r.endswith(".txt") for r in results)

    def test_find_directories_only(self, tmp_path):
        """Test finding directories only."""
        from mngs.io._path import find

        # Create test structure
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()
        (tmp_path / "dir1" / "subdir").mkdir()
        (tmp_path / "file.txt").touch()

        # Find all directories
        results = find(str(tmp_path), type="d", exp="*")

        assert len(results) == 3
        assert all(os.path.isdir(r) for r in results)

    def test_find_with_pattern(self, tmp_path):
        """Test finding with specific pattern."""
        from mngs.io._path import find

        # Create mixed files
        (tmp_path / "test1.py").touch()
        (tmp_path / "test2.py").touch()
        (tmp_path / "data.txt").touch()
        (tmp_path / "script.sh").touch()

        # Find only Python files
        results = find(str(tmp_path), type="f", exp="*.py")

        assert len(results) == 2
        assert all(r.endswith(".py") for r in results)

    def test_find_multiple_patterns(self, tmp_path):
        """Test finding with multiple patterns."""
        from mngs.io._path import find

        # Create various files
        (tmp_path / "test.py").touch()
        (tmp_path / "data.txt").touch()
        (tmp_path / "image.jpg").touch()

        # Find Python and text files
        results = find(str(tmp_path), type="f", exp=["*.py", "*.txt"])

        assert len(results) == 2
        assert any(r.endswith(".py") for r in results)
        assert any(r.endswith(".txt") for r in results)

    def test_find_all_types(self, tmp_path):
        """Test finding both files and directories."""
        from mngs.io._path import find

        # Create mixed structure
        (tmp_path / "file.txt").touch()
        (tmp_path / "directory").mkdir()

        # Find all (type=None)
        results = find(str(tmp_path), type=None, exp="*")

        assert len(results) == 2

    def test_find_recursive(self, tmp_path):
        """Test recursive finding."""
        from mngs.io._path import find

        # Create nested structure
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "level2" / "deep.txt").touch()
        (tmp_path / "shallow.txt").touch()

        # Find should be recursive by default
        results = find(str(tmp_path), type="f", exp="*.txt")

        assert len(results) == 2
        assert any("deep.txt" in r for r in results)
        assert any("shallow.txt" in r for r in results)


class TestFindLatest:
    """Test find_latest functionality."""

    def test_find_latest_basic(self, tmp_path):
        """Test finding latest version of file."""
        from mngs.io._path import find_latest

        # Create versioned files
        (tmp_path / "report_v1.txt").touch()
        (tmp_path / "report_v2.txt").touch()
        (tmp_path / "report_v10.txt").touch()
        (tmp_path / "report_v3.txt").touch()

        result = find_latest(str(tmp_path), "report", ".txt")

        assert result is not None
        assert result.endswith("report_v10.txt")

    def test_find_latest_custom_prefix(self, tmp_path):
        """Test finding latest with custom version prefix."""
        from mngs.io._path import find_latest

        # Create files with custom prefix
        (tmp_path / "data-ver1.csv").touch()
        (tmp_path / "data-ver2.csv").touch()
        (tmp_path / "data-ver5.csv").touch()

        result = find_latest(str(tmp_path), "data", ".csv", version_prefix="-ver")

        assert result is not None
        assert result.endswith("data-ver5.csv")

    def test_find_latest_no_matches(self, tmp_path):
        """Test when no matching files exist."""
        from mngs.io._path import find_latest

        # Create non-matching files
        (tmp_path / "other.txt").touch()
        (tmp_path / "report.txt").touch()  # No version number

        result = find_latest(str(tmp_path), "report", ".txt")

        assert result is None

    def test_find_latest_special_characters(self, tmp_path):
        """Test with special characters in filename."""
        from mngs.io._path import find_latest

        # Create files with special characters
        (tmp_path / "data.backup_v1.tar.gz").touch()
        (tmp_path / "data.backup_v2.tar.gz").touch()

        result = find_latest(str(tmp_path), "data.backup", ".tar.gz")

        assert result is not None
        assert result.endswith("data.backup_v2.tar.gz")

    def test_find_latest_zero_padding(self, tmp_path):
        """Test with zero-padded version numbers."""
        from mngs.io._path import find_latest

        # Create files with zero-padded versions
        (tmp_path / "doc_v001.pdf").touch()
        (tmp_path / "doc_v002.pdf").touch()
        (tmp_path / "doc_v010.pdf").touch()

        result = find_latest(str(tmp_path), "doc", ".pdf")

        assert result is not None
        assert result.endswith("doc_v010.pdf")


class TestPathIntegration:
    """Test integration scenarios."""

    def test_combined_workflow(self, tmp_path):
        """Test a combined workflow using multiple functions."""
        from mngs.io._path import touch, find, split_fpath

        # Create test structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Touch some files
        files = ["analysis_v1.py", "analysis_v2.py", "readme.txt"]
        for f in files:
            touch(str(data_dir / f))

        # Find Python files
        py_files = find(str(data_dir), type="f", exp="*.py")
        assert len(py_files) == 2

        # Split paths
        for py_file in py_files:
            dirname, fname, ext = split_fpath(py_file)
            assert ext == ".py"
            assert "analysis" in fname

    def test_unicode_handling(self, tmp_path):
        """Test handling of Unicode in paths."""
        from mngs.io._path import touch, find, split_fpath

        # Create file with Unicode name
        unicode_file = tmp_path / "文档_v1.txt"
        touch(str(unicode_file))

        # Find and verify
        results = find(str(tmp_path), type="f", exp="*.txt")
        assert len(results) == 1

        # Split path
        dirname, fname, ext = split_fpath(results[0])
        assert "文档_v1" in fname
        assert ext == ".txt"


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
<<<<<<< HEAD
=======

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_path.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 23:18:40 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_path.py
# 
# import fnmatch
# import os
# import re
# from glob import glob as _glob
# 
# from ..path._split import split
# from ..path._this_path import this_path
# 
# ################################################################################
# ## PATH
# ################################################################################
# # def get_this_fpath(when_ipython="/tmp/fake.py"):
# #     """
# #     Get the file path of the calling script, with special handling for IPython environments.
# 
# #     Parameters:
# #     -----------
# #     when_ipython : str, optional
# #         The file path to return when running in an IPython environment. Default is "/tmp/fake.py".
# 
# #     Returns:
# #     --------
# #     str
# #         The file path of the calling script or the specified path for IPython environments.
# 
# #     Example:
# #     --------
# #     >>> import mngs.io._path as path
# #     >>> fpath = path.get_this_fpath()
# #     >>> print(fpath)
# #     '/path/to/current/script.py'
# #     """
# #     THIS_FILE = inspect.stack()[1].filename
# #     if "ipython" in __file__:  # for ipython
# #         THIS_FILE = when_ipython  # "/tmp/fake.py"
# #     return __file__
# 
# 
# # def mk_spath(sfname, makedirs=False):
# #     """
# #     Create a save path based on the calling script's location.
# 
# #     Parameters:
# #     -----------
# #     sfname : str
# #         The name of the file to be saved.
# #     makedirs : bool, optional
# #         If True, create the directory structure for the save path. Default is False.
# 
# #     Returns:
# #     --------
# #     str
# #         The full save path for the file.
# 
# #     Example:
# #     --------
# #     >>> import mngs.io._path as path
# #     >>> spath = path.mk_spath('output.txt', makedirs=True)
# #     >>> print(spath)
# #     '/path/to/current/script/output.txt'
# #     """
# #     THIS_FILE = inspect.stack()[1].filename
# #     if "ipython" in __file__:  # for ipython
# #         THIS_FILE = f'/tmp/fake-{os.getenv("USER")}.py'
# 
# #     ## spath
# #     fpath = __file__
# #     fdir, fname, _ = split_fpath(fpath)
# #     sdir = fdir + fname + "/"
# #     spath = sdir + sfname
# 
# #     if makedirs:
# #         os.makedirs(split(spath)[0], exist_ok=True)
# 
# #     return spath
# 
# 
# def find_the_git_root_dir():
#     """
#     Find the root directory of the current Git repository.
# 
#     Returns:
#     --------
#     str
#         The path to the root directory of the current Git repository.
# 
#     Raises:
#     -------
#     git.exc.InvalidGitRepositoryError
#         If the current directory is not part of a Git repository.
# 
#     Example:
#     --------
#     >>> import mngs.io._path as path
#     >>> git_root = path.find_the_git_root_dir()
#     >>> print(git_root)
#     '/path/to/git/repository'
#     """
#     import git
# 
#     repo = git.Repo(".", search_parent_directories=True)
#     return repo.working_tree_dir
# 
# 
# def split_fpath(fpath):
#     """
#     Split a file path into directory path, file name, and file extension.
# 
#     Parameters:
#     -----------
#     fpath : str
#         The full file path to split.
# 
#     Returns:
#     --------
#     tuple
#         A tuple containing (dirname, fname, ext), where:
#         - dirname: str, the directory path
#         - fname: str, the file name without extension
#         - ext: str, the file extension
# 
#     Example:
#     --------
#     >>> dirname, fname, ext = split_fpath('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
#     >>> print(dirname)
#     '../data/01/day1/split_octave/2kHz_mat/'
#     >>> print(fname)
#     'tt8-2'
#     >>> print(ext)
#     '.mat'
#     """
#     dirname = os.path.dirname(fpath) + "/"
#     base = os.path.basename(fpath)
#     fname, ext = os.path.splitext(base)
#     return dirname, fname, ext
# 
# 
# def touch(fpath):
#     """
#     Create a file or update its modification time.
# 
#     This function mimics the Unix 'touch' command.
# 
#     Parameters:
#     -----------
#     fpath : str
#         The path to the file to be touched.
# 
#     Returns:
#     --------
#     None
# 
#     Side Effects:
#     -------------
#     Creates a new file if it doesn't exist, or updates the modification time
#     of an existing file.
# 
#     Example:
#     --------
#     >>> import mngs.io._path as path
#     >>> import os
#     >>> test_file = '/tmp/test_file.txt'
#     >>> path.touch(test_file)
#     >>> assert os.path.exists(test_file)
#     >>> print(f"File created: {test_file}")
#     File created: /tmp/test_file.txt
#     """
#     import pathlib
# 
#     return pathlib.Path(fpath).touch()
# 
# 
# def find(rootdir, type="f", exp=["*"]):
#     """
#     Mimic the Unix find command to search for files or directories.
# 
#     Parameters:
#     -----------
#     rootdir : str
#         The root directory to start the search from.
#     type : str, optional
#         The type of entries to search for. 'f' for files, 'd' for directories,
#         or None for both. Default is 'f'.
#     exp : str or list of str, optional
#         Pattern(s) to match against file or directory names. Default is ["*"].
# 
#     Returns:
#     --------
#     list
#         A list of paths that match the search criteria.
# 
#     Example:
#     --------
#     >>> find('/path/to/search', "f", "*.txt")
#     ['/path/to/search/file1.txt', '/path/to/search/subdir/file2.txt']
#     """
#     if isinstance(exp, str):
#         exp = [exp]
# 
#     matches = []
#     for _exp in exp:
#         for root, dirs, files in os.walk(rootdir):
#             # Depending on the type, choose the list to iterate over
#             if type == "f":  # Files only
#                 names = files
#             elif type == "d":  # Directories only
#                 names = dirs
#             else:  # All entries
#                 names = files + dirs
# 
#             for name in names:
#                 # Construct the full path
#                 path = os.path.join(root, name)
# 
# 
#                 # If an _exp is provided, use fnmatch to filter names
#                 if _exp and not fnmatch.fnmatch(name, _exp):
#                     continue
# 
#                 # If type is set, ensure the type matches
#                 if type == "f" and not os.path.isfile(path):
#                     continue
#                 if type == "d" and not os.path.isdir(path):
#                     continue
# 
#                 # Add the matching path to the results
#                 matches.append(path)
# 
#     return matches
# 
# 
# def find_latest(dirname, fname, ext, version_prefix="_v"):
#     """
#     Find the latest version of a file with a specific naming pattern.
# 
#     This function searches for files in the given directory that match the pattern:
#     {fname}{version_prefix}{number}{ext} and returns the path of the file with the highest version number.
# 
#     Parameters:
#     -----------
#     dirname : str
#         The directory to search in.
#     fname : str
#         The base filename without version number or extension.
#     ext : str
#         The file extension, including the dot (e.g., '.txt').
#     version_prefix : str, optional
#         The prefix used before the version number. Default is '_v'.
# 
#     Returns:
#     --------
#     str or None
#         The full path of the latest version file if found, None otherwise.
# 
#     Example:
#     --------
#     >>> find_latest('/path/to/dir', 'myfile', '.txt')
#     '/path/to/dir/myfile_v3.txt'
#     """
#     version_pattern = re.compile(
#         rf"({re.escape(fname)}{re.escape(version_prefix)})(\d+)({re.escape(ext)})$"
#     )
# 
#     glob_pattern = os.path.join(dirname, f"{fname}{version_prefix}*{ext}")
#     files = _glob(glob_pattern)
# 
#     highest_version = 0
#     latest_file = None
# 
#     for file in files:
#         filename = os.path.basename(file)
#         match = version_pattern.search(filename)
#         if match:
#             version_num = int(match.group(2))
#             if version_num > highest_version:
#                 highest_version = version_num
#                 latest_file = file
# 
#     return latest_file
# 
# 
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_path.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
