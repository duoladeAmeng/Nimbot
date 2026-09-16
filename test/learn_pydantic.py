"""
Pydantic v2 包教包会 —— 边看边跑，每条 print 都是一个知识点。
用法：在项目根目录执行  uv run python test/learn_pydantic.py
     或者切到 test 目录执行  uv run python learn_pydantic.py
"""

import sys
from datetime import datetime, date
from typing import Annotated, Any
from uuid import UUID, uuid4

# ── 1. BaseModel：一切的基础 ──────────────────────────────────────
print("=" * 60)
print("1. BaseModel —— 定义数据模型")
print("=" * 60)

from pydantic import BaseModel, Field, field_validator, model_validator

class Person:
    name:str
    age:int

p=Person()
p.name=234
print(p.name)


class User(BaseModel):
    """每一行就是一个字段。Pydantic 用类型注解决定如何校验和转换。"""
    name: str
    age: int
    email: str


# 正常构造
u = User(name="小明", age=28, email="xiaoming@example.com")
print(u)  # name='小明' age=28 email='xiaoming@example.com'

# 类型强制转换：str "25" → int 25
u2 = User(name="小红", age="25", email="xiaohong@example.com")
print(u2.age, type(u2.age))  # 25 <class 'int'>

# 转成字典 / JSON
print(u.model_dump())          # {'name': '小明', 'age': 28, 'email': ...}
print(u.model_dump_json())     # JSON 字符串

# 从字典构造
u3 = User.model_validate({"name": "小刚", "age": 30, "email": "g@x.com"})
print(u3)

# ❌ 校验失败会报 ValidationError
try:
    User(name="bad", age="不是数字", email="x@x.com")
except Exception as e:
    print(f"❌ 校验失败（age 必须是数字）: {e}")


# ── 2. Field：字段约束 & 元信息 ───────────────────────────────────
print("\n" + "=" * 60)
print("2. Field —— 默认值、校验约束、别名")
print("=" * 60)


class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100, description="商品名称")
    price: float = Field(gt=0, description="价格，必须大于 0")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    sku: str = Field(alias="productCode")  # 从 JSON 的 productCode 字段取值


p = Product(name="机械键盘", price=399.0, productCode="KB-001")
print(p)              # name='机械键盘' price=399.0 tags=[] sku='KB-001'
print(p.sku)          # KB-001
print(p.model_dump(by_alias=True))  # {'name': ..., 'productCode': 'KB-001', ...}

try:
    Product(name="", price=-5, productCode="X")
except Exception as e:
    print(f"❌ 校验失败: {e}")


# ── 3. 常用字段类型一览 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 常用类型 —— 不只有 str / int")
print("=" * 60)

from pydantic import (
    EmailStr,       # 需要 pip install pydantic[email]
    HttpUrl,
    PositiveInt,
    conint,         # constrained int
    confloat,
    PastDate,
    FutureDate,
)


# 如果你的环境中装了 email-validator，EmailStr 才可用；没装的话先用 str
try:
    class MiscDemo(BaseModel):
        email: EmailStr = "test@example.com"
except ImportError:
    print("(EmailStr 需要 email-validator，用普通 str 演示)")

    class MiscDemo(BaseModel):
        email: str = "test@example.com"


class OrderDemo(BaseModel):
    order_id: UUID = Field(default_factory=uuid4)
    amount: PositiveInt
    url: HttpUrl
    created: date
    score: conint(ge=0, le=100) = 0          # 0~100 的整数


o = OrderDemo(
    amount=100,
    url="https://shop.example.com/item/1",
    created="2025-01-15",                 # str → date 自动转换
    score=95,
)
print(o)
print(o.order_id)  # UUID 自动生成


# ── 4. 嵌套模型 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 嵌套模型 —— 模型套模型")
print("=" * 60)


class Address(BaseModel):
    city: str
    street: str
    zip_code: str


class Employee(BaseModel):
    name: str
    age: int
    address: Address              # 子模型
    tags: list[str] = []


emp = Employee(
    name="张三",
    age=32,
    address={"city": "上海", "street": "南京路", "zip_code": "200000"},
    tags=["engineering", "python"],
)
print(emp)
print(emp.address.city)  # 上海 —— 链式访问


# ── 5. field_validator：单字段校验 ─────────────────────────────────
print("\n" + "=" * 60)
print("5. field_validator —— 给字段加自定义校验逻辑")
print("=" * 60)


class Account(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_no_space(cls, v: str) -> str:
        """用户名不能含空格"""
        if " " in v:
            raise ValueError("用户名不能包含空格")
        return v.strip()

    @field_validator("password", mode="after")  # mode="before" 是转换前执行
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("密码至少 6 位")
        return v


print(Account(username="alice", password="secret123"))  # OK
try:
    Account(username="bad user", password="123")
except Exception as e:
    print(f"❌ 校验失败: {e}")


# ── 6. model_validator：跨字段校验 ─────────────────────────────────
print("\n" + "=" * 60)
print("6. model_validator —— 跨字段联合校验")
print("=" * 60)


class RegisterForm(BaseModel):
    password: str
    password_confirm: str

    @model_validator(mode="after")
    def passwords_match(self) -> "RegisterForm":
        if self.password != self.password_confirm:
            raise ValueError("两次密码不一致")
        return self


try:
    RegisterForm(password="abc123", password_confirm="abc124")
except Exception as e:
    print(f"❌ 校验失败: {e}")

print(RegisterForm(password="abc123", password_confirm="abc123"))  # OK


# ── 7. 配置：model_config ────────────────────────────────────────
print("\n" + "=" * 60)
print("7. model_config —— 全局行为开关")
print("=" * 60)


class StrictUser(BaseModel):
    """不允许自动类型转换"""
    model_config = {"strict": True}

    name: str
    age: int


try:
    StrictUser(name="test", age="25")  # "25" 不会转 int
except Exception as e:
    print(f"❌ strict 模式下 str 不能转 int: {e}")


class FrozenConfig(BaseModel):
    """创建后不可修改（immutable）"""
    model_config = {"frozen": True}

    key: str
    value: int


fc = FrozenConfig(key="mode", value=1)
try:
    fc.value = 2
except Exception as e:
    print(f"❌ frozen 模型不可修改: {e}")


class ExtraForbid(BaseModel):
    """拒绝未知字段（默认是 ignore，会直接丢弃）"""
    model_config = {"extra": "forbid"}

    name: str


try:
    ExtraForbid(name="test", unknown_field=123)
except Exception as e:
    print(f"❌ 不允许额外字段: {e}")


# ── 8. 泛型模型（TypeAdapter）─────────────────────────────────────
print("\n" + "=" * 60)
print("8. TypeAdapter —— 不需要定义 class，直接校验容器类型")
print("=" * 60)

from pydantic import TypeAdapter

# 校验 list[int]
IntList = TypeAdapter(list[int])
print(IntList.validate_python([1, 2, 3]))      # [1, 2, 3]
print(IntList.validate_python(["1", "2"]))      # [1, 2]  自动转换
try:
    IntList.validate_python(["a", "b"])
except Exception as e:
    print(f"❌ 不能转 int: {e}")

# 校验 dict[str, int]
ScoreMap = TypeAdapter(dict[str, int])
print(ScoreMap.validate_python({"math": 90, "english": "85"}))

# 生成 JSON Schema
print(IntList.json_schema())  # 标准 JSON Schema


# ── 9. discriminated union：根据 tag 区分联合类型 ─────────────────
print("\n" + "=" * 60)
print("9. Discriminated Union —— 按标签分发不同类型")
print("=" * 60)

from typing import Literal, Union
from pydantic import BaseModel, Field


class Cat(BaseModel):
    pet_type: Literal["cat"]
    meows: bool


class Dog(BaseModel):
    pet_type: Literal["dog"]
    barks: bool


class Owner(BaseModel):
    name: str
    pet: Cat | Dog = Field(discriminator="pet_type")


# Pydantic 根据 pet_type 自动选 Cat 还是 Dog
owner_cat = Owner(name="小明", pet={"pet_type": "cat", "meows": True})
owner_dog = Owner(name="小红", pet={"pet_type": "dog", "barks": True})
print(owner_cat)  # pet=Cat(pet_type='cat', meows=True)
print(owner_dog)  # pet=Dog(pet_type='dog', barks=True)

# 类型判断
print(type(owner_cat.pet))  # <class '__main__.Cat'>
print(type(owner_dog.pet))  # <class '__main__.Dog'>


# ── 10. computed_field：计算字段 ──────────────────────────────────
print("\n" + "=" * 60)
print("10. computed_field —— 由其他字段计算得出，不出现在 __init__ 里")
print("=" * 60)

from pydantic import computed_field


class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height


r = Rectangle(width=10, height=5)
print(r)          # width=10.0 height=5.0 area=50.0
print(r.area)     # 50.0
print(r.model_dump())  # {'width': 10.0, 'height': 5.0, 'area': 50.0}


# ── 11. 常用生命周期：model_post_init ─────────────────────────────
print("\n" + "=" * 60)
print("11. model_post_init —— 所有字段校验通过后执行的钩子")
print("=" * 60)


class LoggedUser(BaseModel):
    name: str
    created_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context) -> None:
        """校验全部通过后触发，可以做日志、额外初始化等"""
        print(f"  [LOG] 用户 {self.name} 创建完毕，时间={self.created_at}")


u = LoggedUser(name="test_user")


# ── 总结 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 你已掌握 Pydantic v2 的核心能力！回顾一下：")
print("=" * 60)
print("""
  1. BaseModel      — 定义模型，自动校验 + 转换
  2. Field          — 约束、默认值、别名
  3. 丰富类型       — UUID / HttpUrl / PositiveInt / conint / date ...
  4. 嵌套模型       — 模型套模型，自动递归校验
  5. field_validator— 单字段自定义校验
  6. model_validator— 跨字段联合校验
  7. model_config   — strict / frozen / extra 等全局开关
  8. TypeAdapter    — 轻量校验，纯类型不用定义 class
  9. Discriminated Union — 按 tag 分发联合类型
 10. computed_field — 计算字段
 11. model_post_init— 校验通过后的钩子
""")
