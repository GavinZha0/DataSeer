#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/4/18
# @File           : validator.py
# @desc           : pydantic validation

import re


def vali_telephone(value: str) -> str:
    """
    phone number validation
    :param value: phone number
    :return: phone number
    """
    if not value or len(value) != 11 or not value.isdigit():
        raise ValueError("invalid phone number")

    regex = r'^1(3\d|4[4-9]|5[0-35-9]|6[67]|7[013-8]|8[0-9]|9[0-9])\d{8}$'

    if not re.match(regex, value):
        raise ValueError("invalid phone number")

    return value


def vali_email(value: str) -> str:
    """
    email validation
    :param value: email
    :return: email
    """
    if not value:
        raise ValueError("invalid email")

    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(regex, value):
        raise ValueError("invalid email")

    return value




