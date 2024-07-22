#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : crud.py
# @IDE            : PyCharm
# @desc           : Data Access Layer

from core.crud import DalBase
from core.exception import CustomException
from . import model, schema
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, false


class OrgDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(OrgDal, self).__init__()
        self.db = db
        self.model = model.SysOrg
        self.schema = schema.Org


class ParamDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(ParamDal, self).__init__()
        self.db = db
        self.model = model.SysParam
        self.schema = schema.Param


class MenuDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(MenuDal, self).__init__()
        self.db = db
        self.model = model.SysMenu
        self.schema = schema.Menu

    async def get_tree_list(self, mode: int) -> list:
        """
        1：获取菜单树列表
        2：获取菜单树选择项，添加/修改菜单时使用
        3：获取菜单树列表，角色添加菜单权限时使用
        :param mode:
        :return:
        """
        if mode == 3:
            sql = select(self.model).where(self.model.disabled == 0, self.model.is_delete == false())
        else:
            sql = select(self.model).where(self.model.is_delete == false())
        queryset = await self.db.scalars(sql)
        datas = list(queryset.all())
        roots = filter(lambda i: not i.parent_id, datas)
        if mode == 1:
            menus = self.generate_tree_list(datas, roots)
        elif mode == 2 or mode == 3:
            menus = self.generate_tree_options(datas, roots)
        else:
            raise CustomException("获取菜单失败，无可用选项", code=400)
        return self.menus_order(menus)

    async def get_routers(self) -> list:
        """
        获取路由表
        declare interface AppCustomRouteRecordRaw extends Omit<RouteRecordRaw, 'meta'> {
            name: string
            meta: RouteMeta
            component: string
            path: string
            redirect: string
            children?: AppCustomRouteRecordRaw[]
        }
        :param user:
        :return:
        """
        sql = select(self.model) \
            .where(self.model.active == 1, self.model.deleted == false())
        queryset = await self.db.scalars(sql)
        datas = list(queryset.all())
        roots = filter(lambda i: not i.pid, datas)
        menus = self.generate_router_tree(datas, roots)
        return menus

    def generate_router_tree(self, menus: list[model.SysMenu], nodes: filter) -> list:
        """
        生成路由树
        :param menus: 总菜单列表
        :param nodes: 节点菜单列表
        :param name: name拼接，切记Name不能重复
        :return:
        """
        data = []
        for root in nodes:
            router = schema.RouterOut.model_validate(root)
            if root.pid is None:
                sons = filter(lambda i: i.pid == root.id, menus)
                router.children = self.generate_router_tree(menus, sons)
            data.append(router.model_dump())
        return data

    def generate_tree_list(self, menus: list[model.SysMenu], nodes: filter) -> list:
        """
        生成菜单树列表
        :param menus: 总菜单列表
        :param nodes: 每层节点菜单列表
        :return:
        """
        data = []
        for root in nodes:
            router = schema.MenuTreeListOut.model_validate(root)
            if root.menu_type == "0" or root.menu_type == "1":
                sons = filter(lambda i: i.parent_id == root.id, menus)
                router.children = self.generate_tree_list(menus, sons)
            data.append(router.model_dump())
        return data

    def generate_tree_options(self, menus: list[model.SysMenu], nodes: filter) -> list:
        """
        生成菜单树选择项
        :param menus:总菜单列表
        :param nodes:每层节点菜单列表
        :return:
        """
        data = []
        for root in nodes:
            router = {"value": root.id, "label": root.title, "order": root.order}
            if root.menu_type == "0" or root.menu_type == "1":
                sons = filter(lambda i: i.parent_id == root.id, menus)
                router["children"] = self.generate_tree_options(menus, sons)
            data.append(router)
        return data

    @classmethod
    def menus_order(cls, datas: list, order: str = "order", children: str = "children") -> list:
        """
        菜单排序
        :param datas:
        :param order:
        :param children:
        :return:
        """
        result = sorted(datas, key=lambda menu: menu[order])
        for item in result:
            if item[children]:
                item[children] = sorted(item[children], key=lambda menu: menu[order])
        return result

    async def delete_datas(self, ids: list[int], v_soft: bool = False, **kwargs) -> None:
        """
        删除多个菜单
        如果存在角色关联则无法删除
        :param ids: 数据集
        :param v_soft: 是否执行软删除
        :param kwargs: 其他更新字段
        :return:
        """
        count = await RoleDal(self.db).get_count(v_join=[["menus"]], v_where=[self.model.id.in_(ids)])
        if count > 0:
            raise CustomException("无法删除存在角色关联的菜单", code=400)
        await super(MenuDal, self).delete_datas(ids, v_soft, **kwargs)


class RoleDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(RoleDal, self).__init__()
        self.db = db
        self.model = model.SysRole
        self.schema = schema.Role


class UserDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(UserDal, self).__init__()
        self.db = db
        self.model = model.SysUser
        self.schema = schema.User
