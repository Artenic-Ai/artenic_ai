"""Tests for dataset commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


class TestDatasetList:
    def test_table(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {
                "id": "ds1",
                "name": "iris",
                "format": "csv",
                "created_at": "2025-01-01",
                "size": 1024,
            },
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["dataset", "list"])
            assert result.exit_code == 0
            assert "iris" in result.output

    def test_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"id": "ds1", "name": "iris"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "dataset", "list"])
            assert result.exit_code == 0
            assert '"iris"' in result.output


class TestDatasetCreate:
    def test_create(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"dataset_id": "ds1"}):
            result = runner.invoke(
                cli,
                ["dataset", "create", "--name", "iris", "--format", "csv"],
            )
            assert result.exit_code == 0
            assert "ds1" in result.stderr

    def test_create_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"dataset_id": "ds1"}):
            result = runner.invoke(
                cli,
                ["--json", "dataset", "create", "--name", "iris", "--format", "csv"],
            )
            assert result.exit_code == 0
            assert '"dataset_id"' in result.output

    def test_create_with_description(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"dataset_id": "ds2"}):
            result = runner.invoke(
                cli,
                [
                    "dataset",
                    "create",
                    "--name",
                    "titanic",
                    "--format",
                    "parquet",
                    "--description",
                    "Titanic survival dataset",
                ],
            )
            assert result.exit_code == 0


class TestDatasetGet:
    def test_get(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ds1", "name": "iris", "format": "csv"}):
            result = runner.invoke(cli, ["dataset", "get", "ds1"])
            assert result.exit_code == 0
            assert "iris" in result.output

    def test_get_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ds1", "name": "iris"}):
            result = runner.invoke(cli, ["--json", "dataset", "get", "ds1"])
            assert result.exit_code == 0
            assert '"iris"' in result.output


class TestDatasetUpdate:
    def test_update_name(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ds1", "name": "new_name"}):
            result = runner.invoke(cli, ["dataset", "update", "ds1", "--name", "new_name"])
            assert result.exit_code == 0
            assert "updated" in result.stderr

    def test_update_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ds1"}):
            result = runner.invoke(
                cli, ["--json", "dataset", "update", "ds1", "--description", "updated desc"]
            )
            assert result.exit_code == 0

    def test_update_full_options(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"id": "ds1"}):
            result = runner.invoke(
                cli,
                [
                    "dataset",
                    "update",
                    "ds1",
                    "--name",
                    "renamed",
                    "--description",
                    "new desc",
                ],
            )
            assert result.exit_code == 0


class TestDatasetDelete:
    def test_delete(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["dataset", "delete", "ds1"])
            assert result.exit_code == 0
            assert "deleted" in result.stderr

    def test_delete_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=None):
            result = runner.invoke(cli, ["--json", "dataset", "delete", "ds1"])
            assert result.exit_code == 0
            assert '"deleted"' in result.output


class TestDatasetUpload:
    def test_upload(self, runner: CliRunner, tmp_path: Any) -> None:
        sample = tmp_path / "data.csv"
        sample.write_text("a,b\n1,2\n")

        with patch("artenic_ai_cli.commands.datasets._async.run_async") as mock_run:
            mock_run.return_value = {"filename": "data.csv", "size": 8}
            result = runner.invoke(cli, ["dataset", "upload", "ds1", str(sample)])
            assert result.exit_code == 0
            assert "Uploaded" in result.stderr

    def test_upload_json(self, runner: CliRunner, tmp_path: Any) -> None:
        sample = tmp_path / "data.csv"
        sample.write_text("a,b\n1,2\n")

        with patch("artenic_ai_cli.commands.datasets._async.run_async") as mock_run:
            mock_run.return_value = {"filename": "data.csv", "size": 8}
            result = runner.invoke(cli, ["--json", "dataset", "upload", "ds1", str(sample)])
            assert result.exit_code == 0
            assert '"filename"' in result.output

    def test_upload_missing_file(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dataset", "upload", "ds1", "/nonexistent/file.csv"])
        assert result.exit_code != 0


class TestDatasetDownload:
    def test_download(self, runner: CliRunner, tmp_path: Any) -> None:
        with patch("artenic_ai_cli.commands.datasets._async.run_async") as mock_run:
            mock_run.return_value = b"col1,col2\n1,2\n"
            dest = tmp_path / "out.csv"
            result = runner.invoke(cli, ["dataset", "download", "ds1", "data.csv", "-o", str(dest)])
            assert result.exit_code == 0
            assert dest.read_bytes() == b"col1,col2\n1,2\n"

    def test_download_json(self, runner: CliRunner, tmp_path: Any) -> None:
        with patch("artenic_ai_cli.commands.datasets._async.run_async") as mock_run:
            mock_run.return_value = b"content"
            dest = tmp_path / "out.bin"
            result = runner.invoke(
                cli, ["--json", "dataset", "download", "ds1", "file.bin", "-o", str(dest)]
            )
            assert result.exit_code == 0
            assert '"downloaded"' in result.output

    def test_download_default_path(
        self, runner: CliRunner, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("artenic_ai_cli.commands.datasets._async.run_async") as mock_run:
            mock_run.return_value = b"data"
            result = runner.invoke(cli, ["dataset", "download", "ds1", "result.csv"])
            assert result.exit_code == 0


class TestDatasetFiles:
    def test_files(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {"filename": "train.csv", "size": 2048, "uploaded_at": "2025-01-01"},
            {"filename": "test.csv", "size": 512, "uploaded_at": "2025-01-02"},
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["dataset", "files", "ds1"])
            assert result.exit_code == 0
            assert "train.csv" in result.output

    def test_files_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"filename": "train.csv"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "dataset", "files", "ds1"])
            assert result.exit_code == 0
            assert '"train.csv"' in result.output


class TestDatasetStats:
    def test_stats(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"rows": 150, "columns": 5, "size_bytes": 4096}):
            result = runner.invoke(cli, ["dataset", "stats", "ds1"])
            assert result.exit_code == 0
            assert "150" in result.output

    def test_stats_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"rows": 150}):
            result = runner.invoke(cli, ["--json", "dataset", "stats", "ds1"])
            assert result.exit_code == 0
            assert "150" in result.output


class TestDatasetPreview:
    def test_preview(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {"col1": "a", "col2": 1},
            {"col1": "b", "col2": 2},
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["dataset", "preview", "ds1"])
            assert result.exit_code == 0

    def test_preview_with_limit(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value=[{"x": 1}]):
            result = runner.invoke(cli, ["dataset", "preview", "ds1", "--limit", "5"])
            assert result.exit_code == 0

    def test_preview_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"col1": "a"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "dataset", "preview", "ds1"])
            assert result.exit_code == 0


class TestDatasetVersionList:
    def test_version_list(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [
            {"version": 1, "message": "initial", "created_at": "2025-01-01"},
            {"version": 2, "message": "update", "created_at": "2025-01-02"},
        ]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["dataset", "version", "list", "ds1"])
            assert result.exit_code == 0
            assert "initial" in result.output

    def test_version_list_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        data = [{"version": 1, "message": "v1"}]
        with patch_run_async(return_value=data):
            result = runner.invoke(cli, ["--json", "dataset", "version", "list", "ds1"])
            assert result.exit_code == 0
            assert '"v1"' in result.output


class TestDatasetVersionCreate:
    def test_version_create(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"version": 3}):
            result = runner.invoke(
                cli, ["dataset", "version", "create", "ds1", "-m", "new snapshot"]
            )
            assert result.exit_code == 0
            assert "3" in result.stderr

    def test_version_create_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"version": 3}):
            result = runner.invoke(
                cli, ["--json", "dataset", "version", "create", "ds1", "-m", "snap"]
            )
            assert result.exit_code == 0
            assert '"version"' in result.output

    def test_version_create_no_message(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"version": 1}):
            result = runner.invoke(cli, ["dataset", "version", "create", "ds1"])
            assert result.exit_code == 0


class TestDatasetLineage:
    def test_lineage(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(
            return_value={"dataset_id": "ds1", "parents": ["ds0"], "children": ["ds2"]}
        ):
            result = runner.invoke(cli, ["dataset", "lineage", "ds1"])
            assert result.exit_code == 0
            assert "ds0" in result.output

    def test_lineage_json(self, runner: CliRunner, patch_run_async: Any) -> None:
        with patch_run_async(return_value={"dataset_id": "ds1", "parents": []}):
            result = runner.invoke(cli, ["--json", "dataset", "lineage", "ds1"])
            assert result.exit_code == 0
            assert '"dataset_id"' in result.output
