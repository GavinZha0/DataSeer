# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/19
# @File           : router.py
# @desc           : app router

from apps.auth.views import app as auth_app
from apps.vis.views import app as vis_app
from apps.ml.views import app as ml_app
from apps.ai.views import app as ai_app


urlpatterns = [
    {"ApiRouter": auth_app, "prefix": "/auth", "tags": ["Authorization"]},
    {"ApiRouter": vis_app, "prefix": "/vis", "tags": ["Data Visualization"]},
    {"ApiRouter": ml_app, "prefix": "/ml", "tags": ["Machine Learning"]},
    {"ApiRouter": ai_app, "prefix": "/ai", "tags": ["Artificial Intelligence"]}
]
