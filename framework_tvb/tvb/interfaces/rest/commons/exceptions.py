# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

from abc import abstractmethod

from tvb.basic.exceptions import TVBException


class BaseRestException(TVBException):
    def __init__(self, message=None, code=None, payload=None):
        Exception.__init__(self)
        self.message = message if message is not None and message.strip() else self.get_default_message()
        self.code = code
        self.payload = payload

    def to_dict(self):
        payload = dict(self.payload or ())
        payload['message'] = self.message
        payload['code'] = self.code
        return payload

    @abstractmethod
    def get_default_message(self):
        return None


class BadRequestException(BaseRestException):
    def __init__(self, message, payload=None):
        super(BadRequestException, self).__init__(message, code=400, payload=payload)

    def get_default_message(self):
        return "Bad request error"


class InvalidIdentifierException(BaseRestException):
    def __init__(self, message=None, payload=None):
        super(InvalidIdentifierException, self).__init__(message, code=404, payload=payload)

    def get_default_message(self):
        return "No data found for the given identifier"


class ClientException(BaseRestException):
    def __init__(self, message, code):
        super(ClientException, self).__init__(message, code)

    def get_default_message(self):
        return "There was an error on client request"