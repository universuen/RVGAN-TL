import context

from src import config
from src.logger import Logger


if __name__ == '__main__':
    config.logger.level = 'DEBUG'
    logger = Logger('test')
    logger.debug('debug test')
    logger.info('info test')
    logger.warning('warning test')
    logger.error('error test')
    logger.critical('critical test')
