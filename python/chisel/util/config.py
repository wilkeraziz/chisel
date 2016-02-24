"""
@author waziz
"""
from chisel.exception import InputError
import logging
from ast import literal_eval
import re


class Config(object):

    SECTION = re.compile(r'\[([^]]+)\](.*)')
    KVPAIR = re.compile(r' *([^ ]+) *= *(.+)')

    def __init__(self, path, as_block=['cdec', 'cdec:features'], required=[]):
        self._path = path
        self._as_block = set(as_block)
        self._required = set(required)
        with open(path, 'r') as fi:
            self._lines = fi.readlines()
        self._i = 0
        self._sections = {}
        self._parse()
        missing = self._required - set(self._sections.iterkeys()) 
        if missing:
            raise InputError(', '.join(missing), 'Missing required sections')

    def write(self, ostream):
        for name, data in sorted(self._sections.iteritems(), key=lambda pair: pair[0]):
            ostream.write('[%s]\n' % name)
            if name in self._as_block:
                for line in data:
                    ostream.write('{0}\n'.format(line))
            else:
                for k, v in data.iteritems():
                    ostream.write('%s = %r\n' % (k, v))
            ostream.write('\n')

    def __str__(self):
        lines = []
        for name, data in sorted(self._sections.iteritems(), key=lambda pair: pair[0]):
            lines.append('[%s]' % name)
            if name in self._as_block:
                lines.extend(data)
            else:
                for k, v in data.iteritems():
                    lines.append('%s = %r' % (k, v))
            lines.append('\n')
        return '\n'.join(lines)

    def _parsing(self):
        return self._i < len(self._lines)
    
    def _next_line(self):
        if self._parsing():
            self._i += 1
            return self._lines[self._i - 1].strip()
        else:
            return None

    def _backtrack(self):
        if self._i > 0:
            self._i -= 1
            return True
        return False

    def _ignore(self, line):
        aux = line.strip()
        return not aux or aux.startswith('#')

    def _parse_as_block(self):
        block = []
        while self._parsing():
            line = self._next_line()
            if self._ignore(line):
                continue
            # try to match a section: in which case the current one is finished
            if Config.SECTION.match(line):
                self._backtrack()
                break
            block.append(line)
        return block

    def _parse_as_dict(self):
        section = {}
        while self._parsing():
            line = self._next_line()
            if self._ignore(line):
                continue
            # try to match a section: in which case the current one is finished
            if Config.SECTION.match(line):
                self._backtrack()
                break
            # try to match a key-value pair
            m = Config.KVPAIR.match(line)
            if m is None:
                raise InputError(line, 'Unexpected format within section')
            if len(m.groups()) != 2:
                raise InputError(line, 'Expected a key-value pair')
            key = m.group(1)
            value = literal_eval(m.group(2))
            section[key] = value
        return section

    def _parse_section(self, header):
        if header in self._as_block:
            self._sections[header] = self._parse_as_block()
        else: # parse as dict
            self._sections[header] = self._parse_as_dict()

    def _parse(self):
        while self._parsing():
            line = self._next_line()
            if self._ignore(line):  # ignore comments and empty lines
                continue 
            # try to match a section
            m = Config.SECTION.match(line)
            if m is None:
                raise InputError(line, 'Unexpected format')
            if len(m.groups()) != 2:
                raise InputError(line, 'Expected a section header (possibly followed by a comment)')
            header = m.group(1)
            tail = m.group(2)
            if not self._ignore(tail):
                raise InputError(line, 'Section headers can only be followed by comments')
            self._parse_section(header)

    def is_block(self, name):
        return name in self._as_block

    def has_section(self, name):
        return name in self._sections

    def items(self, name):
        if name in self._as_block:
            return tuple(self._sections[name])
        else:
            return self._sections[name].items()

    def add_section(self, name, block=False):
        if name in self._sections:
            return False
        if block:
            self._as_block.add(name)
            self._sections[name] = []
        else:
            self._sections[name] = {}
        return True

    def set(self, section, key, value=None):
        if section in self._as_block:
            self._sections[section].append(key)
        else:
            self._sections[section][key] = value

    def remove_section(self, section):
        if section in self._sections:
            del self._sections[section]
            if section in self._as_block:
                self._as_block.remove(section)


def configure(parser, set_defaults=[], required_sections=['proxy', 'target', 'cdec'], configure_logging=True):
    """

    :param argparse.ArgumentParser parser:
    :return:
    """
    args = parser.parse_args()

    if configure_logging:
        if args.verbose:
            if args.verbose > 1:
                logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
            else:
                logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    
    
    config = Config(args.config, required=required_sections)

    # some command line options may be overwritten by the section 'chisel:sampler' in the config file
    for section in set_defaults:
        if not config.has_section(section):
            continue
        if config.is_block(section):
            logging.info('Cannot overwrite defaults for block section (%s): skipping it', section)
            continue

        options = dict(config.items(section))
        logging.debug('set_defaults [%s]: %s', section, options)
        parser.set_defaults(**options)
        # reparse options (with new defaults) TODO: find a better way
        args = parser.parse_args()

    return args, config, False
