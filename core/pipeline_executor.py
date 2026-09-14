import os
import json
import shutil
import subprocess
import logging
import contextlib
import io
import platform

from pathlib import Path

from core.utils import TranslationConfig
from ocr_modules import unlimited_ocr
from ocr_modules.translate_md_with_glassory import TranslationPipeline
from ocr_modules.translated_json_to_md_unlimited_ocr import run_json_to_md
from core.db import update_job_status


logger = logging.getLogger("pipeline_executor")


# ============================================================
# PRINT / STDOUT / STDERR -> LOGGER
# ============================================================

@contextlib.contextmanager
def redirect_prints_to_logger(
    logger_instance: logging.Logger,
    prefix: str = ""
):
    """
    Redirect print(), stdout and stderr output from imported functions
    to the existing application logger.

    This allows functions such as:

        unlimited_ocr.process_pdf()
        TranslationPipeline.run()
        run_json_to_md()

    to remain unchanged while their console output is written
    to the pipeline logger.
    """

    class LoggerWriter(io.TextIOBase):

        def __init__(
            self,
            log: logging.Logger,
            level: int,
            prefix: str = ""
        ):
            self.log = log
            self.level = level
            self.prefix = prefix

        def write(self, message):
            if not message:
                return 0

            # print() may call write() multiple times.
            # Remove trailing newline and log each actual line.
            message = message.rstrip()

            if message:
                for line in message.splitlines():
                    line = line.strip()

                    if line:
                        self.log.log(
                            self.level,
                            f"{self.prefix}{line}"
                        )

            return len(message)

        def flush(self):
            pass

        def isatty(self):
            return False

    stdout_writer = LoggerWriter(
        logger_instance,
        logging.INFO,
        prefix=prefix
    )

    stderr_writer = LoggerWriter(
        logger_instance,
        logging.ERROR,
        prefix=prefix
    )

    with contextlib.redirect_stdout(stdout_writer), \
         contextlib.redirect_stderr(stderr_writer):

        yield


# ============================================================
# PANDOC
# ============================================================

def get_pandoc_executable() -> str:
    """
    Find and validate the Pandoc executable.

    Works on both Windows and Linux/Unix.
    """

    system = platform.system()

    # --------------------------------------------------------
    # 1. Try PATH first
    # --------------------------------------------------------

    pandoc_cmd = shutil.which("pandoc")

    # --------------------------------------------------------
    # 2. Windows fallback
    # --------------------------------------------------------

    if not pandoc_cmd and system == "Windows":

        windows_path = Path(
            r"C:\Program Files\Pandoc\pandoc.exe"
        )

        if windows_path.exists():
            pandoc_cmd = str(windows_path)

    # --------------------------------------------------------
    # 3. Linux / Unix fallback
    # --------------------------------------------------------

    if not pandoc_cmd and system != "Windows":

        unix_path = Path("/usr/local/bin/pandoc")

        if unix_path.exists():
            pandoc_cmd = str(unix_path)

    # --------------------------------------------------------
    # 4. Make sure a path was found
    # --------------------------------------------------------

    if not pandoc_cmd:
        raise RuntimeError(
            "Pandoc executable was not found in PATH "
            "or in the expected installation directory."
        )

    pandoc_path = Path(pandoc_cmd)

    # --------------------------------------------------------
    # 5. Check existence
    # --------------------------------------------------------

    if not pandoc_path.exists():

        raise RuntimeError(
            f"Pandoc executable does not exist: "
            f"{pandoc_path}"
        )

    # --------------------------------------------------------
    # 6. Check file
    # --------------------------------------------------------

    if not pandoc_path.is_file():

        raise RuntimeError(
            f"Pandoc path is not a file: "
            f"{pandoc_path}"
        )

    # --------------------------------------------------------
    # 7. Linux / Unix executable permission
    # --------------------------------------------------------

    if system != "Windows":

        if not os.access(pandoc_path, os.X_OK):

            raise RuntimeError(
                f"Pandoc exists but is not executable: "
                f"{pandoc_path}"
            )

    # --------------------------------------------------------
    # 8. Actually execute Pandoc
    # --------------------------------------------------------

    try:

        result = subprocess.run(
            [
                str(pandoc_path),
                "--version"
            ],
            capture_output=True,
            text=True,
            check=True
        )

    except PermissionError as exc:

        raise RuntimeError(
            f"Pandoc exists but permission is denied: "
            f"{pandoc_path}"
        ) from exc

    except (
        OSError,
        subprocess.SubprocessError
    ) as exc:

        raise RuntimeError(
            f"Pandoc executable cannot be executed: "
            f"{pandoc_path}. "
            f"Error: {exc}"
        ) from exc

    # --------------------------------------------------------
    # 9. Log version
    # --------------------------------------------------------

    version = (
        result.stdout.splitlines()[0]
        if result.stdout
        else "unknown"
    )

    logger.info(
        f"Pandoc found: {pandoc_path}"
    )

    logger.info(
        f"Pandoc version: {version}"
    )

    return str(pandoc_path)


# ============================================================
# FILE VALIDATION
# ============================================================

def is_file_valid(file_path: Path) -> bool:
    """
    Checks if a file exists and is not empty.

    This protects against resuming from a 0-byte file
    if the process crashed during writing.
    """

    return (
        file_path.exists()
        and file_path.is_file()
        and file_path.stat().st_size > 0
    )


# ============================================================
# PANDOC EXECUTION
# ============================================================

def run_pandoc_conversion(
    pandoc_cmd: str,
    input_path: Path,
    output_path: Path,
    reference_doc: str,
    job_id
):
    """
    Execute Pandoc and redirect its stdout/stderr
    to the existing application logger.
    """

    command = [
        pandoc_cmd,
        str(input_path),
        "-o",
        str(output_path)
    ]

    # Add reference document only if configured.
    if reference_doc:

        command.extend(
            [
                "--reference-doc",
                str(reference_doc)
            ]
        )

    logger.info(
        f"[{job_id}] [PANDOC] Starting conversion..."
    )

    logger.info(
        f"[{job_id}] [PANDOC] Input: {input_path}"
    )

    logger.info(
        f"[{job_id}] [PANDOC] Output: {output_path}"
    )

    if reference_doc:

        logger.info(
            f"[{job_id}] [PANDOC] Reference: {reference_doc}"
        )

    logger.info(
        f"[{job_id}] [PANDOC] Executable: {pandoc_cmd}"
    )

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

    except PermissionError as exc:

        raise RuntimeError(
            f"Pandoc permission denied: {pandoc_cmd}"
        ) from exc

    except OSError as exc:

        raise RuntimeError(
            f"Could not execute Pandoc: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Pandoc stdout
    # --------------------------------------------------------

    if result.stdout:

        for line in result.stdout.splitlines():

            line = line.strip()

            if line:

                logger.info(
                    f"[{job_id}] [PANDOC] {line}"
                )

    # --------------------------------------------------------
    # Pandoc stderr
    # --------------------------------------------------------

    if result.stderr:

        for line in result.stderr.splitlines():

            line = line.strip()

            if line:

                if result.returncode == 0:

                    logger.warning(
                        f"[{job_id}] [PANDOC] {line}"
                    )

                else:

                    logger.error(
                        f"[{job_id}] [PANDOC] {line}"
                    )

    # --------------------------------------------------------
    # Check exit code
    # --------------------------------------------------------

    if result.returncode != 0:

        raise RuntimeError(
            f"Pandoc failed with exit code "
            f"{result.returncode}: "
            f"{result.stderr}"
        )

    logger.info(
        f"[{job_id}] [PANDOC] Conversion completed successfully."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(job: dict):

    job_id = job["JobID"]

    input_filepath = Path(
        job["InputFilePath"]
    )

    job_dir = input_filepath.parent

    stem = input_filepath.stem

    # ========================================================
    # FILE PATHS
    # ========================================================

    ocr_json_path = (
        job_dir / f"{stem}.json"
    )

    persian_json_path = (
        job_dir / f"{stem}_persian.json"
    )

    persian_md_path = (
        job_dir / f"{stem}_persian.md"
    )

    docx_path = (
        job_dir / f"{stem}_translated.docx"
    )

    images_dir = (
        job_dir / f"{stem}_images"
    )

    debug_dir = (
        job_dir / f"{stem}_debug"
    )

    temp_md = (
        job_dir / f"{stem}_temp.md"
    )

    try:

        logger.info(
            f"[{job_id}] ========================================"
        )

        logger.info(
            f"[{job_id}] Starting translation pipeline"
        )

        logger.info(
            f"[{job_id}] Input file: {input_filepath}"
        )

        logger.info(
            f"[{job_id}] Working directory: {job_dir}"
        )

        # ====================================================
        # 1. OCR STAGE
        # ====================================================

        if not is_file_valid(ocr_json_path):

            logger.info(
                f"[{job_id}] Starting OCR..."
            )

            update_job_status(
                job_id,
                "OCR_PROCESSING",
                status_detail="در حال OCR فایل..."
            )

            # ------------------------------------------------
            # Patch globals for unlimited_ocr
            # ------------------------------------------------

            unlimited_ocr.PDF_PATH = (
                str(input_filepath)
            )

            unlimited_ocr.OUTPUT_MD = (
                str(temp_md)
            )

            unlimited_ocr.OUTPUT_JSON = (
                str(ocr_json_path)
            )

            unlimited_ocr.IMAGES_DIR = (
                str(images_dir)
            )

            unlimited_ocr.DEBUG_DIR = (
                str(debug_dir)
            )

            logger.info(
                f"[{job_id}] [OCR] PDF_PATH="
                f"{input_filepath}"
            )

            logger.info(
                f"[{job_id}] [OCR] OUTPUT_JSON="
                f"{ocr_json_path}"
            )

            logger.info(
                f"[{job_id}] [OCR] Starting process_pdf()..."
            )

            # ------------------------------------------------
            # Capture all print()/stdout/stderr from OCR
            # ------------------------------------------------

            with redirect_prints_to_logger(
                logger,
                prefix=f"[{job_id}] [OCR] "
            ):

                result = unlimited_ocr.process_pdf(
                    str(input_filepath),
                    job_id=job_id
                )

            logger.info(
                f"[{job_id}] [OCR] process_pdf() completed."
            )

            # ------------------------------------------------
            # Validate result
            # ------------------------------------------------

            if result is None:

                raise RuntimeError(
                    "OCR returned None."
                )

            if "json" not in result:

                raise RuntimeError(
                    "OCR result does not contain "
                    "'json' field."
                )

            # ------------------------------------------------
            # Save OCR JSON
            # ------------------------------------------------

            with open(
                ocr_json_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    result["json"],
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            logger.info(
                f"[{job_id}] [OCR] JSON saved: "
                f"{ocr_json_path}"
            )

        else:

            logger.info(
                f"[{job_id}] ✅ OCR stage already "
                f"completed. Skipping..."
            )

        # ====================================================
        # 2. TRANSLATION STAGE
        # ====================================================

        if not is_file_valid(
            persian_json_path
        ):

            update_job_status(
                job_id,
                "TRANSLATING",
                status_detail="در حال ترجمه فایل..."
            )

            logger.info(
                f"[{job_id}] Starting Translation..."
            )

            # ------------------------------------------------
            # Create translation pipeline
            # ------------------------------------------------

            pipeline = TranslationPipeline(

                input_path=str(
                    ocr_json_path
                ),

                output_path=str(
                    persian_json_path
                ),

                workers_merge_headers=4,

                workers_merge_paragraphs=4,

                workers_translate_headings=4,

                workers_translate_toc=4,

                workers_translate_body=3,

                job_id=job_id
            )

            logger.info(
                f"[{job_id}] [TRANSLATION] "
                f"TranslationPipeline created."
            )

            logger.info(
                f"[{job_id}] [TRANSLATION] "
                f"Starting pipeline.run()..."
            )

            # ------------------------------------------------
            # Capture print()/stdout/stderr
            # ------------------------------------------------

            with redirect_prints_to_logger(
                logger,
                prefix=f"[{job_id}] [TRANSLATION] "
            ):

                pipeline.run()

            logger.info(
                f"[{job_id}] [TRANSLATION] "
                f"pipeline.run() completed."
            )

            # ------------------------------------------------
            # Validate output
            # ------------------------------------------------

            if not is_file_valid(
                persian_json_path
            ):

                raise RuntimeError(
                    "Translation completed but "
                    "Persian JSON file was not created "
                    "or is empty."
                )

            logger.info(
                f"[{job_id}] [TRANSLATION] "
                f"Output: {persian_json_path}"
            )

        else:

            logger.info(
                f"[{job_id}] ✅ Translation stage "
                f"already completed. Skipping..."
            )

        # ====================================================
        # 3. JSON -> MARKDOWN
        # ====================================================

        if not is_file_valid(
            persian_md_path
        ):

            update_job_status(
                job_id,
                "MD_CONVERSION",
                status_detail=(
                    "در حال تبدیل محتوای ترجمه شده "
                    "به فرمت Markdown ..."
                )
            )

            logger.info(
                f"[{job_id}] Starting JSON to MD..."
            )

            logger.info(
                f"[{job_id}] [MD] Input JSON: "
                f"{persian_json_path}"
            )

            logger.info(
                f"[{job_id}] [MD] Output MD: "
                f"{persian_md_path}"
            )

            # ------------------------------------------------
            # Capture print()/stdout/stderr
            # ------------------------------------------------

            with redirect_prints_to_logger(
                logger,
                prefix=f"[{job_id}] [MD] "
            ):

                run_json_to_md(

                    json_path=str(
                        persian_json_path
                    ),

                    pdf_path=str(
                        input_filepath
                    ),

                    dpi=600,

                    output_path=str(
                        persian_md_path
                    ),

                    lang="fa"
                )

            logger.info(
                f"[{job_id}] [MD] "
                f"run_json_to_md() completed."
            )

            # ------------------------------------------------
            # Validate output
            # ------------------------------------------------

            if not is_file_valid(
                persian_md_path
            ):

                raise RuntimeError(
                    "JSON to Markdown conversion "
                    "completed but Markdown file "
                    "was not created or is empty."
                )

            logger.info(
                f"[{job_id}] [MD] Output created: "
                f"{persian_md_path}"
            )

        else:

            logger.info(
                f"[{job_id}] ✅ MD Conversion stage "
                f"already completed. Skipping..."
            )

        # ====================================================
        # 4. MARKDOWN -> DOCX
        # ====================================================

        if not is_file_valid(
            docx_path
        ):

            update_job_status(
                job_id,
                "DOCX_CONVERSION",
                status_detail=(
                    "در حال تبدیل فایل Markdown "
                    "به Docx..."
                )
            )

            logger.info(
                f"[{job_id}] Starting Pandoc "
                f"DOCX conversion..."
            )

            # ------------------------------------------------
            # Find Pandoc
            # ------------------------------------------------

            pandoc_cmd = (
                get_pandoc_executable()
            )

            # ------------------------------------------------
            # Reference DOCX
            # ------------------------------------------------

            pandoc_ref_path = getattr(
                TranslationConfig,
                "LOCAL_PANDOC_DOCX_PATH",
                ""
            )

            # ------------------------------------------------
            # Validate reference document
            # ------------------------------------------------

            if pandoc_ref_path:

                reference_path = Path(
                    pandoc_ref_path
                )

                if not reference_path.exists():

                    raise RuntimeError(
                        "Pandoc reference document "
                        f"does not exist: "
                        f"{reference_path}"
                    )

                if not reference_path.is_file():

                    raise RuntimeError(
                        "Pandoc reference document "
                        f"is not a file: "
                        f"{reference_path}"
                    )

                logger.info(
                    f"[{job_id}] [PANDOC] "
                    f"Reference document validated: "
                    f"{reference_path}"
                )

            else:

                logger.info(
                    f"[{job_id}] [PANDOC] "
                    f"No reference document configured."
                )

            # ------------------------------------------------
            # Run Pandoc
            # ------------------------------------------------

            run_pandoc_conversion(

                pandoc_cmd=pandoc_cmd,

                input_path=persian_md_path,

                output_path=docx_path,

                reference_doc=pandoc_ref_path,

                job_id=job_id
            )

            # ------------------------------------------------
            # Validate DOCX
            # ------------------------------------------------

            if not is_file_valid(
                docx_path
            ):

                raise RuntimeError(
                    "Pandoc reported success but "
                    "the output DOCX file was not "
                    "created or is empty."
                )

            logger.info(
                f"[{job_id}] [PANDOC] "
                f"DOCX created: {docx_path}"
            )

        else:

            logger.info(
                f"[{job_id}] ✅ DOCX Conversion stage "
                f"already completed. Skipping..."
            )

        # ====================================================
        # 5. CLEANUP
        # ====================================================

        logger.info(
            f"[{job_id}] All pipeline stages completed."
        )

        logger.info(
            f"[{job_id}] Cleaning up intermediate files..."
        )

        # ----------------------------------------------------
        # Remove intermediate files
        # ----------------------------------------------------

        intermediate_files = [

            ocr_json_path,

            persian_json_path,

            persian_md_path,

            temp_md
        ]

        for file_path in intermediate_files:

            try:

                if file_path.exists():

                    file_path.unlink()

                    logger.info(
                        f"[{job_id}] Removed: "
                        f"{file_path}"
                    )

            except Exception as exc:

                # Cleanup failure should not cause
                # an otherwise successful translation
                # to become FAILED.
                logger.warning(
                    f"[{job_id}] Could not remove "
                    f"{file_path}: {exc}"
                )

        # ----------------------------------------------------
        # Remove directories
        # ----------------------------------------------------

        intermediate_dirs = [

            images_dir,

            debug_dir
        ]

        for directory in intermediate_dirs:

            try:

                if directory.exists():

                    shutil.rmtree(
                        directory
                    )

                    logger.info(
                        f"[{job_id}] Removed directory: "
                        f"{directory}"
                    )

            except Exception as exc:

                logger.warning(
                    f"[{job_id}] Could not remove "
                    f"directory {directory}: {exc}"
                )

        # ====================================================
        # 6. COMPLETED
        # ====================================================

        update_job_status(

            job_id,

            "COMPLETED",

            status_detail=(
                "فایل ترجمه شده با فرمت "
                "Docx آماده دانلود است."
            ),

            output_filepath=str(
                docx_path
            )
        )

        logger.info(
            f"[{job_id}] 🎉 Successfully completed."
        )

        logger.info(
            f"[{job_id}] Output: {docx_path}"
        )

        logger.info(
            f"[{job_id}] ========================================"
        )

    except Exception as exc:

        # ====================================================
        # FAILURE
        # ====================================================

        logger.exception(
            f"[{job_id}] ❌ Pipeline failed at current step"
        )

        # IMPORTANT:
        # Do NOT clean up intermediate files here.
        #
        # This allows the next execution to resume from
        # the last successfully generated intermediate file.

        try:

            update_job_status(
                job_id,
                "FAILED",
                error_message=str(exc)
            )

        except Exception:

            logger.exception(
                f"[{job_id}] Failed to update "
                f"job status to FAILED."
            )

        logger.error(
            f"[{job_id}] Pipeline stopped."
        )