
import logging


RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


class Log:
    def __init__(self):
        self.log = logging.getLogger("MultiLabelModel")
        
    def h1(self, text: str) -> None:
        line = "=" * 60
        self.log.info(f"\n{BLUE}{line}")
        self.log.info(text.upper())
        self.log.info(f"{line}{RESET}\n")

    def h2(self, text: str) -> None:
        line = "-" * 40
        self.log.info(f"{CYAN}{line}")
        self.log.info(text)
        self.log.info(f"{line}{RESET}")

    def body(self, text: str) -> None:
        self.log.info(text)

    def info(self, text: str) -> None:
        self.log.info(f"{GREEN}[INFO]{RESET} {text}")

    def warn(self, text: str) -> None:
        self.log.warning(f"{YELLOW}[WARN]{RESET} {text}")
    
    def error(self, text: str) -> None:
        self.log.error(f"{RED}[ERROR]{RESET} {text}")
