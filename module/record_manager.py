import json
import sqlite3
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import pathlib


# =============================================================================
# 表名：sale_records
# 说明：门店每日销售记录数据，按字段存储
# 字段：id, upc, date, move, price, customer_count, created_at
# 索引：idx_sale_sku_date(upc, date)


# 表名：review_records
# 说明：商品评价记录，按字段存储
# 字段：id, record_id, upc, date, rating, comment, category, dimension, merchandise_id, supplier_id, created_at
# 索引：idx_review_upc_date(upc, date), idx_review_id_unique(record_id)

# 表名：return_rate_records
# 说明：商品退货率记录
# 字段：id, sku_id, return_rate, return_number, date, created_at
# 索引：idx_return_rate_sku_date(sku_id, date)

# 表名：new_records
# 说明：通用内容记录，按字段存储
# 字段：id, record_id(唯一), title, content, created_at
# 索引：idx_new_id_unique(record_id)

# =============================================================================
# 表名：reviews（未来需要评价数据时启用）
# 说明：商品评价信息
# 字段：
#   id       : 主键
#   upc      : SKU 标识
#   date     : 评论日期
#   rating   : 评分
#   comment  : 评论内容
# 用途：
#   - 商品质量与消费者反馈分析
#   - 可与销售数据结合看是否影响转化率
# =============================================================================
class ReviewRecord:
    """
    标准化的一条商品评价记录，date 为 datetime.date 对象
    """
    def __init__(
        self,
        record_id: str,
        upc: str,
        date_obj: date,
        rating: int,
        comment: str,
        category: Optional[str] = None,
        dimension: Optional[str] = None,
        merchandise_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ):
        if not isinstance(date_obj, date):
            raise TypeError("date must be a datetime.date object")
        self.id = record_id
        self.upc = upc
        self.date = date_obj
        self.rating = rating
        self.comment = comment
        self.category = category
        self.dimension = dimension
        self.merchandise_id = merchandise_id
        self.supplier_id = supplier_id

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "upc": self.upc,
            "date": self.date.isoformat(),
            "rating": self.rating,
            "comment": self.comment,
            "category": self.category,
            "dimension": self.dimension,
            "merchandise_id": self.merchandise_id,
            "supplier_id": self.supplier_id,
        }

    @staticmethod
    def from_dict(data: Dict) -> "ReviewRecord":
        return ReviewRecord(
            record_id=data["id"],
            upc=data["upc"],
            date_obj=datetime.strptime(data["date"], "%Y-%m-%d").date(),
            rating=data["rating"],
            comment=data["comment"],
            category=data.get("category"),
            dimension=data.get("dimension", ""),
            merchandise_id=data.get("merchandise_id"),
            supplier_id=data.get("supplier_id"),
        )


class ReturnRateRecord:
    """
    SKU 退货率记录。
    """
    def __init__(self, sku_id: str, return_rate: float, return_number: int, date_obj: date):
        if not isinstance(date_obj, date):
            raise TypeError("date must be a datetime.date object")
        self.sku_id = sku_id
        self.return_rate = return_rate
        self.return_number = return_number
        self.date = date_obj

    def to_dict(self) -> Dict:
        return {
            "sku_id": self.sku_id,
            "return_rate": self.return_rate,
            "return_number": self.return_number,
            "date": self.date.isoformat(),
        }

    @staticmethod
    def from_dict(data: Dict) -> "ReturnRateRecord":
        return ReturnRateRecord(
            sku_id=data["sku_id"],
            return_rate=data["return_rate"],
            return_number=data.get("return_number", 0),
            date_obj=datetime.strptime(data["date"], "%Y-%m-%d").date(),
        )


class ReturnRecord:
    """
    退货记录：记录每次退货的详细信息
    - supplier_id: 供给商 id
    - sku_id: 退货的 SKU
    - date: 退货日期
    """
    def __init__(self, supplier_id: str, sku_id: str, date_obj: date):
        if not isinstance(date_obj, date):
            raise TypeError("date must be a datetime.date object")
        self.supplier_id = supplier_id
        self.sku_id = sku_id
        self.date = date_obj

    def to_dict(self) -> Dict:
        return {
            "supplier_id": self.supplier_id,
            "sku_id": self.sku_id,
            "date": self.date.isoformat(),
        }

    @staticmethod
    def from_dict(data: Dict) -> "ReturnRecord":
        return ReturnRecord(
            supplier_id=data["supplier_id"],
            sku_id=data["sku_id"],
            date_obj=datetime.strptime(data["date"], "%Y-%m-%d").date(),
        )


class SaleRecord:
    """
    标准化的一条订单记录（门店销售），date 为 datetime.date 对象
    """
    def __init__(self, upc: str, date_obj: date, move: int, price: float, customer_count: Optional[int] = None):
        if not isinstance(date_obj, date):
            raise TypeError("date must be a datetime.date object")
        self.upc = upc
        self.date = date_obj
        self.move = move
        self.price = price
        self.customer_count = customer_count

    def to_dict(self) -> Dict:
        """用于存数据库（序列化为 JSON）"""
        return {
            "upc": self.upc,
            "date": self.date.isoformat(),  # 转为 YYYY-MM-DD
            "move": self.move,
            "price": self.price,
            "customer_count": self.customer_count
        }

    @staticmethod
    def from_dict(data: Dict) -> "SaleRecord":
        """用于从数据库反序列化"""
        return SaleRecord(
            upc=data["upc"],
            date_obj=datetime.strptime(data["date"], "%Y-%m-%d").date(),
            move=data["move"],
            price=data["price"],
            customer_count=data.get("customer_count")
        )


class NewRecord:
    """
    通用数据内容记录
    """
    def __init__(self, record_id: str, news_id: str, title: str, content: str, date_obj: date):
        if not isinstance(date_obj, date):
            raise TypeError("date must be a datetime.date object")
        self.id = record_id
        self.news_id = news_id
        self.title = title
        self.content = content
        self.date = date_obj

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "news_id": self.news_id,
            "title": self.title,
            "content": self.content,
            "date": self.date.isoformat(),
        }

    @staticmethod
    def from_dict(data: Dict) -> "NewRecord":
        return NewRecord(
            record_id=data["id"],
            news_id=data.get("news_id"),
            title=data["title"],
            content=data["content"],
            date_obj=datetime.strptime(data["date"], "%Y-%m-%d").date(),
        )


# ======================= 新增：供给商价格记录 =======================

class SupplierPriceRecord:
    """
    供给商价格记录:
    - supplier_id: 供给商 id
    - sku_id: 供给商提供的某个 sku
    - date: 日期
    - price: 该日期供给商提供的价格
    """
    def __init__(self, supplier_id: str, sku_id: str, date_obj: date, price: float):
        if not isinstance(date_obj, date):
            raise TypeError("date must be a datetime.date object")
        self.supplier_id = supplier_id
        self.sku_id = sku_id
        self.date = date_obj
        self.price = price

    def to_dict(self) -> Dict:
        return {
            "supplier_id": self.supplier_id,
            "sku_id": self.sku_id,
            "date": self.date.isoformat(),
            "price": self.price,
        }

    @staticmethod
    def from_dict(data: Dict) -> "SupplierPriceRecord":
        return SupplierPriceRecord(
            supplier_id=data["supplier_id"],
            sku_id=data["sku_id"],
            date_obj=datetime.strptime(data["date"], "%Y-%m-%d").date(),
            price=data["price"],
        )


class SupplierOrderRecord:
    """
    给供给商的订单记录:
    - supplier_id: 哪个供给商
    - order_date: 下单日期
    - arrival_date: 到达日期
    - shipping_days: 运输时间（天）
    - items: 下单的商品对象 { SKU: number }
    - cost: 订单总成本
    """
    def __init__(
        self,
        supplier_id: str,
        order_date: date,
        items: Dict[str, int],
        arrival_date: Optional[date] = None,
        shipping_days: Optional[int] = None,
        cost: float = 0.0,
    ):
        if not isinstance(order_date, date):
            raise TypeError("order_date must be a datetime.date object")
        if arrival_date is not None and not isinstance(arrival_date, date):
            raise TypeError("arrival_date must be a datetime.date object or None")

        self.supplier_id = supplier_id
        self.order_date = order_date
        self.arrival_date = arrival_date
        self.items = items or {}

        # 如果没传运输时间但有到达时间，则自动计算
        if shipping_days is None and arrival_date is not None:
            shipping_days = (arrival_date - order_date).days
        self.shipping_days = shipping_days
        self.cost = cost

    def to_dict(self) -> Dict:
        return {
            "supplier_id": self.supplier_id,
            "order_date": self.order_date.isoformat(),
            "arrival_date": self.arrival_date.isoformat() if self.arrival_date else None,
            "shipping_days": self.shipping_days,
            "items": self.items,
            "cost": self.cost,
        }

    @staticmethod
    def from_dict(data: Dict) -> "SupplierOrderRecord":
        order_date = datetime.strptime(data["order_date"], "%Y-%m-%d").date()
        arrival_raw = data.get("arrival_date")
        arrival_date = (
            datetime.strptime(arrival_raw, "%Y-%m-%d").date()
            if arrival_raw is not None
            else None
        )
        return SupplierOrderRecord(
            supplier_id=data["supplier_id"],
            order_date=order_date,
            arrival_date=arrival_date,
            shipping_days=data.get("shipping_days"),
            items=data.get("items") or {},
            cost=data.get("cost", 0.0),
        )


class ProductLifecycleRecord:
    """
    商品生命周期记录：追踪每个商品从购入到售出/损耗的完整过程。
    - merch_id: 商品唯一ID
    - status: in_stock / sold / expired
    """
    def __init__(
        self,
        merch_id: str,
        sku_id: str,
        supplier_id: str,
        order_id: str,
        buy_price: float,
        order_date: date,
        arrival_date: date,
        expired_time: date,
        selling_price: Optional[float] = None,
        sold_date: Optional[date] = None,
        loss_date: Optional[date] = None,
        status: str = "in_stock",
    ):
        self.merch_id = merch_id
        self.sku_id = sku_id
        self.supplier_id = supplier_id
        self.order_id = order_id
        self.buy_price = buy_price
        self.order_date = order_date
        self.arrival_date = arrival_date
        self.expired_time = expired_time
        self.selling_price = selling_price
        self.sold_date = sold_date
        self.loss_date = loss_date
        self.status = status

    def to_dict(self) -> Dict:
        return {
            "merch_id": self.merch_id,
            "sku_id": self.sku_id,
            "supplier_id": self.supplier_id,
            "order_id": self.order_id,
            "buy_price": self.buy_price,
            "selling_price": self.selling_price,
            "order_date": self.order_date.isoformat(),
            "arrival_date": self.arrival_date.isoformat(),
            "expired_time": self.expired_time.isoformat(),
            "sold_date": self.sold_date.isoformat() if self.sold_date else None,
            "loss_date": self.loss_date.isoformat() if self.loss_date else None,
            "status": self.status,
        }

    @staticmethod
    def from_dict(data: Dict) -> "ProductLifecycleRecord":
        return ProductLifecycleRecord(
            merch_id=data["merch_id"],
            sku_id=data["sku_id"],
            supplier_id=data["supplier_id"],
            order_id=data.get("order_id"),
            buy_price=data["buy_price"],
            order_date=datetime.strptime(data["order_date"], "%Y-%m-%d").date(),
            arrival_date=datetime.strptime(data["arrival_date"], "%Y-%m-%d").date(),
            expired_time=datetime.strptime(data["expired_time"], "%Y-%m-%d").date(),
            selling_price=data.get("selling_price"),
            sold_date=datetime.strptime(data["sold_date"], "%Y-%m-%d").date() if data.get("sold_date") else None,
            loss_date=datetime.strptime(data["loss_date"], "%Y-%m-%d").date() if data.get("loss_date") else None,
            status=data.get("status", "in_stock"),
        )


class RecordManager:
    """
    Order record manager that organizes data by SKU and stores records in a database.
    """

    def __init__(self, 
            data_dir: str = "order_records",
            init_sql_path: Optional[str] = None,
        ):
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "records.db"

        # 新增：可选的 SQL 初始化文件
        self.init_sql_path: Optional[pathlib.Path] = (
            pathlib.Path(init_sql_path) if init_sql_path is not None else None
        )

        self._init_db()

    def _get_table_columns(self, cursor, table: str) -> set[str]:
        cursor.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cursor.fetchall()}

    def _ensure_column(self, cursor, table: str, column: str, definition: str) -> None:
        """Add column if missing; used for lightweight in-place schema upgrades."""
        cols = self._get_table_columns(cursor, table)
        if column not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _init_db(self):
        db_file = pathlib.Path(self.db_path)
        if db_file.exists():
            db_file.unlink()  # 删除文件
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if self.init_sql_path is not None:
            if not self.init_sql_path.exists():
                conn.close()
                raise FileNotFoundError(f"init_sql file not found: {self.init_sql_path}")

            sql_text = self.init_sql_path.read_text(encoding="utf-8")
            # executescript 支持多条语句（含 ; 换行）
            cursor.executescript(sql_text)
            # 轻量升级：确保新增列存在
            try:
                self._ensure_column(cursor, "review_records", "dimension", "TEXT")
                self._ensure_column(cursor, "review_records", "category", "TEXT")
                self._ensure_column(cursor, "review_records", "merchandise_id", "TEXT")
                self._ensure_column(cursor, "review_records", "supplier_id", "TEXT")
                self._ensure_column(cursor, "supplier_orders", "cost", "REAL DEFAULT 0")
                # return_rate_records 可能不存在，确保创建并包含 return_number 列
                cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS return_rate_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sku_id TEXT NOT NULL,
                        return_rate REAL NOT NULL,
                        return_number INTEGER NOT NULL DEFAULT 0,
                        date TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    '''
                )
                self._ensure_column(cursor, "return_rate_records", "return_number", "INTEGER NOT NULL DEFAULT 0")
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_return_rate_sku_date ON return_rate_records (sku_id, date)')
                # return_records 表
                cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS return_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        supplier_id TEXT NOT NULL,
                        sku_id TEXT NOT NULL,
                        date TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    '''
                )
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_return_supplier_sku_date ON return_records (supplier_id, sku_id, date)')
                self._ensure_column(cursor, "new_records", "date", "TEXT")
                self._ensure_column(cursor, "new_records", "news_id", "TEXT")
                # product_lifecycle 表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS product_lifecycle (
                        merch_id TEXT PRIMARY KEY,
                        sku_id TEXT NOT NULL,
                        supplier_id TEXT NOT NULL,
                        order_id TEXT,
                        buy_price REAL NOT NULL,
                        selling_price REAL,
                        order_date TEXT NOT NULL,
                        arrival_date TEXT NOT NULL,
                        expired_time TEXT NOT NULL,
                        sold_date TEXT,
                        loss_date TEXT,
                        status TEXT NOT NULL DEFAULT 'in_stock',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_lifecycle_sku_status ON product_lifecycle (sku_id, status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_lifecycle_sku_status_sold_date ON product_lifecycle (sku_id, status, sold_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_lifecycle_order ON product_lifecycle (order_id)')
            except Exception:
                pass

        else:
            # ================== sale_records ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sale_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upc TEXT NOT NULL,
                    date TEXT NOT NULL,
                    move INTEGER NOT NULL,
                    price REAL NOT NULL,
                    customer_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sale_sku_date ON sale_records (upc, date)')

            # ================== review_records ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS review_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    upc TEXT NOT NULL,
                    date TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT,
                    category TEXT,
                    dimension TEXT,
                    merchandise_id TEXT,
                    supplier_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_review_upc_date ON review_records (upc, date)')
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_review_id_unique ON review_records (record_id)')

            # ================== return_rate_records ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS return_rate_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku_id TEXT NOT NULL,
                    return_rate REAL NOT NULL,
                    return_number INTEGER NOT NULL DEFAULT 0,
                    date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_return_rate_sku_date ON return_rate_records (sku_id, date)')

            # ================== return_records ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS return_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    supplier_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_return_supplier_sku_date ON return_records (supplier_id, sku_id, date)')

            # ================== new_records ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS new_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    news_id TEXT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_new_id_unique ON new_records (record_id)')

            # ================== 新增：供给商价格表 ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS supplier_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    supplier_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    price REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_supplier_price
                ON supplier_prices (supplier_id, sku_id, date)
            ''')

            # ================== 新增：供给订单表 ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS supplier_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    supplier_id TEXT NOT NULL,
                    order_date TEXT NOT NULL,
                    arrival_date TEXT,
                    shipping_days INTEGER,
                    cost REAL DEFAULT 0,
                    items TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_supplier_order_date
                ON supplier_orders (supplier_id, order_date)
            ''')

            # ================== 新增：商品生命周期表 ==================
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS product_lifecycle (
                    merch_id TEXT PRIMARY KEY,
                    sku_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL,
                    order_id TEXT,
                    buy_price REAL NOT NULL,
                    selling_price REAL,
                    order_date TEXT NOT NULL,
                    arrival_date TEXT NOT NULL,
                    expired_time TEXT NOT NULL,
                    sold_date TEXT,
                    loss_date TEXT,
                    status TEXT NOT NULL DEFAULT 'in_stock',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lifecycle_sku_status ON product_lifecycle (sku_id, status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lifecycle_sku_status_sold_date ON product_lifecycle (sku_id, status, sold_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lifecycle_order ON product_lifecycle (order_id)')

            conn.commit()
            conn.close()

    @staticmethod
    def _normalize_date(d) -> Optional[str]:
        """
        d 可以是 datetime.date 或 'YYYY-MM-DD'
        返回标准字符串 YYYY-MM-DD
        """
        if d is None:
            return None
        if isinstance(d, date):
            return d.isoformat()
        if isinstance(d, str):
            datetime.strptime(d, "%Y-%m-%d")  # 校验格式
            return d
        raise TypeError("Date must be datetime.date or 'YYYY-MM-DD' string")

    # ================== 门店销售记录 ==================

    def add_record(self, sku_id: str, record: SaleRecord):
        """
        保存门店销售 Record；sku_id 需与 record.upc 一致
        """
        if record.upc != sku_id:
            raise ValueError("sku_id must match record.upc for sale_records")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT INTO sale_records (upc, date, move, price, customer_count)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (
                record.upc or sku_id,
                record.date.isoformat(),
                record.move,
                record.price,
                record.customer_count,
            )
        )

        conn.commit()
        conn.close()

    def read_sku(
        self,
        sku_id: str,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None
    ) -> Dict[str, List[SaleRecord]]:
        """
        返回指定 SKU/UPC 的所有 Record（按日期范围过滤）
        """
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conditions = ["upc = ?"]
        params: List[str] = [sku_id]
        if start_str is not None:
            conditions.append("date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("date <= ?")
            params.append(end_str)

        where_clause = " AND ".join(conditions)
        cursor.execute(
            f'''
            SELECT date, upc, move, price, customer_count
            FROM sale_records
            WHERE {where_clause}
            ORDER BY date, id
            ''',
            tuple(params),
        )

        rows = cursor.fetchall()
        conn.close()


        result: Dict[str, List[SaleRecord]] = {}
        for date_str, upc, move, price, customer_count in rows:
            record_dict = {
                "upc": upc,
                "date": date_str,
                "move": move,
                "price": price,
                "customer_count": customer_count,
            }
            record = SaleRecord.from_dict(record_dict)

            if date_str not in result:
                result[date_str] = []
            result[date_str].append(record)

        return result

    # ================== 评论记录 ==================

    def add_review(self, review: ReviewRecord) -> None:
        """
        保存一条评论记录。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT OR IGNORE INTO review_records (
                record_id,
                upc,
                date,
                rating,
                comment,
                category,
                dimension,
                merchandise_id,
                supplier_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                review.id,
                review.upc,
                review.date.isoformat(),
                review.rating,
                review.comment,
                review.category,
                review.dimension,
                review.merchandise_id,
                review.supplier_id,
            ),
        )

        conn.commit()
        conn.close()

    def add_news_record(self, record: NewRecord) -> None:
        """保存一条通用新闻记录。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT OR IGNORE INTO new_records (record_id, news_id, title, content, date)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (record.id, record.news_id, record.title, record.content, record.date.isoformat()),
        )
        conn.commit()
        conn.close()

    def read_news(
        self,
        news_id: Optional[str] = None,
        record_id: Optional[str] = None,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
        limit: Optional[int] = None,
    ) -> List[NewRecord]:
        """
        读取新闻记录，可按 news_id/record_id 和日期范围过滤。
        """
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        conditions = []
        params: List[Any] = []
        if news_id:
            conditions.append("news_id = ?")
            params.append(news_id)
        if record_id:
            conditions.append("record_id = ?")
            params.append(record_id)
        if start_str is not None:
            conditions.append("date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("date <= ?")
            params.append(end_str)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT record_id, news_id, title, content, date, created_at
            FROM new_records
            {where_clause}
            ORDER BY date ASC, created_at ASC
            {limit_clause}
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        conn.close()

        records: List[NewRecord] = []
        for rec_id, n_id, title, content, date_str, _created in rows:
            data = {
                "id": rec_id,
                "news_id": n_id or rec_id,
                "title": title,
                "content": content,
                "date": date_str,
            }
            records.append(NewRecord.from_dict(data))
        return records

    def read_reviews(
        self,
        sku_id: str,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> List[ReviewRecord]:
        """
        按 SKU 查询评论，可选日期区间。
        """
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        available_cols = self._get_table_columns(cursor, "review_records")
        select_cols = ["record_id", "upc", "date", "rating", "comment"]
        has_category = "category" in available_cols
        has_dimension = "dimension" in available_cols
        has_merch_id = "merchandise_id" in available_cols
        has_supplier_id = "supplier_id" in available_cols
        if has_category:
            select_cols.append("category")
        if has_dimension:
            select_cols.append("dimension")
        if has_merch_id:
            select_cols.append("merchandise_id")
        if has_supplier_id:
            select_cols.append("supplier_id")

        conditions = ["upc = ?"]
        params: List[str] = [sku_id]
        if start_str is not None:
            conditions.append("date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("date <= ?")
            params.append(end_str)

        where_clause = " AND ".join(conditions)
        sql = f'''
            SELECT {", ".join(select_cols)}
            FROM review_records
            WHERE {where_clause}
            ORDER BY date, id
        '''

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        results: List[ReviewRecord] = []
        for row in rows:
            row_dict = dict(zip(select_cols, row))
            record_dict = {
                "id": row_dict.get("record_id"),
                "upc": row_dict.get("upc"),
                "date": row_dict.get("date"),
                "rating": row_dict.get("rating"),
                "comment": row_dict.get("comment"),
                "category": row_dict.get("category") if has_category else None,
                "dimension": row_dict.get("dimension") if has_dimension else None,
                "merchandise_id": row_dict.get("merchandise_id") if has_merch_id else None,
                "supplier_id": row_dict.get("supplier_id") if has_supplier_id else None,
            }
            results.append(ReviewRecord.from_dict(record_dict))

        return results

    # ================== 退货率记录 ==================

    def add_return_rate(self, record: "ReturnRateRecord") -> None:
        """保存一条退货率记录。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO return_rate_records (sku_id, return_rate, return_number, date)
            VALUES (?, ?, ?, ?)
            ''',
            (
                record.sku_id,
                record.return_rate,
                record.return_number,
                record.date.isoformat(),
            )
        )
        conn.commit()
        conn.close()

    def read_return_rates(
        self,
        sku_id: str,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> List["ReturnRateRecord"]:
        """查询退货率记录，按日期范围过滤。"""
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conditions = ["sku_id = ?"]
        params: List[str] = [sku_id]
        if start_str is not None:
            conditions.append("date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("date <= ?")
            params.append(end_str)

        where_clause = " AND ".join(conditions)
        cursor.execute(
            f'''
            SELECT sku_id, return_rate, return_number, date
            FROM return_rate_records
            WHERE {where_clause}
            ORDER BY date ASC, id ASC
            ''',
            tuple(params),
        )
        rows = cursor.fetchall()
        conn.close()

        results: List[ReturnRateRecord] = []
        for sku, rate, num, date_str in rows:
            data = {"sku_id": sku, "return_rate": rate, "return_number": num, "date": date_str}
            results.append(ReturnRateRecord.from_dict(data))
        return results

    # ================== 退货记录操作 ==================

    def add_return(self, record: ReturnRecord) -> None:
        """保存一条退货记录。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO return_records (supplier_id, sku_id, date)
            VALUES (?, ?, ?)
            ''',
            (
                record.supplier_id,
                record.sku_id,
                record.date.isoformat(),
            )
        )
        conn.commit()
        conn.close()

    def read_returns(
        self,
        supplier_id: Optional[str] = None,
        sku_id: Optional[str] = None,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> List[ReturnRecord]:
        """查询退货记录，可按 supplier_id / sku_id / 日期范围过滤。"""
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conditions = []
        params: List = []

        if supplier_id is not None:
            conditions.append("supplier_id = ?")
            params.append(supplier_id)
        if sku_id is not None:
            conditions.append("sku_id = ?")
            params.append(sku_id)
        if start_str is not None:
            conditions.append("date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("date <= ?")
            params.append(end_str)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f'''
            SELECT supplier_id, sku_id, date
            FROM return_records
            {where_clause}
            ORDER BY supplier_id, sku_id, date, id
        '''

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        result: List[ReturnRecord] = []
        for supplier, sku, date_str in rows:
            record_dict = {
                "supplier_id": supplier,
                "sku_id": sku,
                "date": date_str,
            }
            result.append(ReturnRecord.from_dict(record_dict))
        return result

    # ================== 供给商价格操作 ==================

    def add_supplier_price(self, record: SupplierPriceRecord):
        """
        保存一条供给商价格记录
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT INTO supplier_prices (supplier_id, sku_id, date, price)
            VALUES (?, ?, ?, ?)
            ''',
            (
                record.supplier_id,
                record.sku_id,
                record.date.isoformat(),
                record.price,
            )
        )

        conn.commit()
        conn.close()

    def read_supplier_prices(
        self,
        supplier_id: Optional[str] = None,
        sku_id: Optional[str] = None,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> List[SupplierPriceRecord]:
        """
        查询供给商价格记录，可按 supplier_id / sku_id / 日期范围过滤
        """
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conditions = []
        params: List = []

        if supplier_id is not None:
            conditions.append("supplier_id = ?")
            params.append(supplier_id)
        if sku_id is not None:
            conditions.append("sku_id = ?")
            params.append(sku_id)
        if start_str is not None:
            conditions.append("date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("date <= ?")
            params.append(end_str)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f'''
            SELECT supplier_id, sku_id, date, price
            FROM supplier_prices
            {where_clause}
            ORDER BY supplier_id, sku_id, date, id
        '''

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        result: List[SupplierPriceRecord] = []
        for supplier, sku, date_str, price in rows:
            record_dict = {
                "supplier_id": supplier,
                "sku_id": sku,
                "date": date_str,
                "price": price,
            }
            result.append(SupplierPriceRecord.from_dict(record_dict))
        return result

    # ================== 供给订单操作 ==================

    def add_supplier_order(self, order: SupplierOrderRecord):
        """
        保存一条给供给商的订单记录
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        items_json = json.dumps(order.items)

        cursor.execute(
            '''
            INSERT INTO supplier_orders (
                supplier_id,
                order_date,
                arrival_date,
                shipping_days,
                cost,
                items
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (
                order.supplier_id,
                order.order_date.isoformat(),
                order.arrival_date.isoformat() if order.arrival_date else None,
                order.shipping_days,
                order.cost,
                items_json,
            )
        )

        conn.commit()
        conn.close()

    def read_supplier_orders(
        self,
        supplier_id: Optional[str] = None,
        start_order_date: Optional[date | str] = None,
        end_order_date: Optional[date | str] = None,
    ) -> List[SupplierOrderRecord]:
        """
        查询供给订单记录，可按 supplier_id + 下单日期范围过滤
        （items 是一个 { SKU: number } 的对象）
        """
        start_str = self._normalize_date(start_order_date)
        end_str = self._normalize_date(end_order_date)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conditions = []
        params: List = []

        if supplier_id is not None:
            conditions.append("supplier_id = ?")
            params.append(supplier_id)
        if start_str is not None:
            conditions.append("order_date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("order_date <= ?")
            params.append(end_str)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f'''
            SELECT supplier_id, order_date, arrival_date, shipping_days, cost, items
            FROM supplier_orders
            {where_clause}
            ORDER BY order_date, id
        '''

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        result: List[SupplierOrderRecord] = []
        for supplier_id, order_date, arrival_date, shipping_days, cost, items_json in rows:
            items = json.loads(items_json) if items_json else {}
            order_dict = {
                "supplier_id": supplier_id,
                "order_date": order_date,
                "arrival_date": arrival_date,
                "shipping_days": shipping_days,
                "items": items,
                "cost": cost,
            }
            result.append(SupplierOrderRecord.from_dict(order_dict))

        return result

    # ================== 商品生命周期操作 ==================

    def add_product_lifecycle(self, record: ProductLifecycleRecord) -> None:
        """保存一条商品生命周期记录。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO product_lifecycle (
                merch_id, sku_id, supplier_id, order_id, buy_price, selling_price,
                order_date, arrival_date, expired_time, sold_date, loss_date, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                record.merch_id,
                record.sku_id,
                record.supplier_id,
                record.order_id,
                record.buy_price,
                record.selling_price,
                record.order_date.isoformat(),
                record.arrival_date.isoformat(),
                record.expired_time.isoformat(),
                record.sold_date.isoformat() if record.sold_date else None,
                record.loss_date.isoformat() if record.loss_date else None,
                record.status,
            ),
        )
        conn.commit()
        conn.close()

    def batch_add_product_lifecycle(self, records: List[ProductLifecycleRecord]) -> None:
        """批量插入商品生命周期记录。"""
        if not records:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany(
            '''
            INSERT INTO product_lifecycle (
                merch_id, sku_id, supplier_id, order_id, buy_price, selling_price,
                order_date, arrival_date, expired_time, sold_date, loss_date, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            [
                (
                    r.merch_id, r.sku_id, r.supplier_id, r.order_id,
                    r.buy_price, r.selling_price,
                    r.order_date.isoformat(), r.arrival_date.isoformat(),
                    r.expired_time.isoformat(),
                    r.sold_date.isoformat() if r.sold_date else None,
                    r.loss_date.isoformat() if r.loss_date else None,
                    r.status,
                )
                for r in records
            ],
        )
        conn.commit()
        conn.close()

    def update_lifecycle_sold(self, merch_id: str, selling_price: float, sold_date: date) -> None:
        """更新商品为已售出状态。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            UPDATE product_lifecycle
            SET status = 'sold', selling_price = ?, sold_date = ?
            WHERE merch_id = ?
            ''',
            (selling_price, sold_date.isoformat(), merch_id),
        )
        conn.commit()
        conn.close()

    def update_lifecycle_expired(self, merch_id: str, loss_date: date) -> None:
        """更新商品为已损耗状态。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            UPDATE product_lifecycle
            SET status = 'expired', loss_date = ?
            WHERE merch_id = ?
            ''',
            (loss_date.isoformat(), merch_id),
        )
        conn.commit()
        conn.close()

    def read_product_lifecycle(
        self,
        sku_id: Optional[str] = None,
        status: Optional[str] = None,
        order_id: Optional[str] = None,
        start_order_date: Optional[date | str] = None,
        end_order_date: Optional[date | str] = None,
        start_sold_date: Optional[date | str] = None,
        end_sold_date: Optional[date | str] = None,
    ) -> List[ProductLifecycleRecord]:
        """查询商品生命周期记录，支持按 sku_id / status / order_id / 下单日期或售出日期范围过滤。"""
        start_str = self._normalize_date(start_order_date)
        end_str = self._normalize_date(end_order_date)
        start_sold_str = self._normalize_date(start_sold_date)
        end_sold_str = self._normalize_date(end_sold_date)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conditions = []
        params: List = []

        if sku_id is not None:
            conditions.append("sku_id = ?")
            params.append(sku_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if order_id is not None:
            conditions.append("order_id = ?")
            params.append(order_id)
        if start_str is not None:
            conditions.append("order_date >= ?")
            params.append(start_str)
        if end_str is not None:
            conditions.append("order_date <= ?")
            params.append(end_str)
        if start_sold_str is not None:
            conditions.append("sold_date >= ?")
            params.append(start_sold_str)
        if end_sold_str is not None:
            conditions.append("sold_date <= ?")
            params.append(end_sold_str)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f'''
            SELECT merch_id, sku_id, supplier_id, order_id, buy_price, selling_price,
                   order_date, arrival_date, expired_time, sold_date, loss_date, status
            FROM product_lifecycle
            {where_clause}
            ORDER BY order_date, merch_id
        '''

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        result: List[ProductLifecycleRecord] = []
        for (
            merch_id, sku, supplier, oid, buy_p, sell_p,
            o_date, a_date, exp_time, s_date, l_date, st
        ) in rows:
            data = {
                "merch_id": merch_id,
                "sku_id": sku,
                "supplier_id": supplier,
                "order_id": oid,
                "buy_price": buy_p,
                "selling_price": sell_p,
                "order_date": o_date,
                "arrival_date": a_date,
                "expired_time": exp_time,
                "sold_date": s_date,
                "loss_date": l_date,
                "status": st,
            }
            result.append(ProductLifecycleRecord.from_dict(data))
        return result

    # ================== 通用 SQL ==================

    def execute_sql(self, sql: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """
        执行自定义 SQL
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if params is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, params)

        rows = cursor.fetchall()
        conn.commit()
        conn.close()
        return rows

    def execute_sql_with_columns(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> tuple[list[str], list[Tuple]]:
        """
        执行 SQL 并返回 (列名列表, 结果行)。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if params is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, params)

        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        conn.commit()
        conn.close()
        return columns, rows
    
    def dump_to_sql(self, output_file: Optional[str | pathlib.Path] = None) -> pathlib.Path:
        """
        将当前 SQLite 数据库中所有表的建表语句 + 数据
        一次性导出到一个 .sql 文件中。

        参数:
            output_file: 导出的目标文件路径。
                         如果为 None，则默认导出到 data_dir / "dump.sql"

        返回:
            导出后的文件路径（pathlib.Path）
        """
        if output_file is None:
            output_path = self.data_dir / "dump.sql"
        else:
            output_path = pathlib.Path(output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        try:
            with output_path.open("w", encoding="utf-8") as f:
                # iterdump 会生成：
                #   - BEGIN TRANSACTION;
                #   - CREATE TABLE ...
                #   - INSERT INTO ...
                #   - COMMIT;
                for line in conn.iterdump():
                    f.write(f"{line}\n")
        finally:
            conn.close()

        return output_path
    
    def restore_from_sql(self, sql_file: pathlib.Path) -> None:
        """
        从 SQL 文件恢复数据库。
        
        参数:
            sql_file: SQL 文件路径
        """
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_file}")
        
        # 删除现有数据库
        db_file = pathlib.Path(self.db_path)
        if db_file.exists():
            db_file.unlink()
        
        # 创建新数据库并执行 SQL
        conn = sqlite3.connect(self.db_path)
        try:
            sql_text = sql_file.read_text(encoding="utf-8")
            cursor = conn.cursor()
            # executescript 支持多条语句（含 ; 换行）
            cursor.executescript(sql_text)
            conn.commit()
        finally:
            conn.close()


# ============================================================
#                     🔥 测试用例（run_tests）
# ============================================================

def run_tests():
    print("===== Running RecordManager Tests =====")
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    print(f"[INFO] Using temporary directory: {temp_dir}")

    rm = RecordManager(data_dir=temp_dir)

    # Test 1: Add + read_sku correctness
    try:
        sku_id = "0001"
        r1 = SaleRecord(sku_id, date(2024, 1, 1), 10, 1.0, 100)
        r2 = SaleRecord(sku_id, date(2024, 1, 1), 20, 1.5, 120)
        r3 = SaleRecord(sku_id, date(2024, 1, 2), 5, 2.0, 80)

        rm.add_record(sku_id, r1)
        rm.add_record(sku_id, r2)
        rm.add_record(sku_id, r3)

        data = rm.read_sku(sku_id)

        assert len(data["2024-01-01"]) == 2
        assert len(data["2024-01-02"]) == 1
        assert isinstance(data["2024-01-01"][0], SaleRecord)
        print("[PASS] Test 1: Adding + read_sku working")
    except Exception as e:
        print("[FAIL] Test 1 error:", e)

    # Test 2: date range filtering
    try:
        sku_id = "B001"

        rm.add_record(sku_id, SaleRecord(sku_id, date(2024, 1, 1), 1, 1.0, 50))
        rm.add_record(sku_id, SaleRecord(sku_id, date(2024, 1, 5), 2, 2.0, 60))
        rm.add_record(sku_id, SaleRecord(sku_id, date(2024, 1, 10), 3, 3.0, 70))

        result = rm.read_sku(sku_id, start_date=date(2024, 1, 2), end_date=date(2024, 1, 7))

        assert "2024-01-05" in result
        assert "2024-01-01" not in result
        assert "2024-01-10" not in result
        print("[PASS] Test 2: Date range filter working")
    except Exception as e:
        print("[FAIL] Test 2 error:", e)

    # Test 3: Wrong date type error
    try:
        # 这里故意把 date 写成字符串，应该触发 TypeError
        bad = SaleRecord("C001", "2024-01-01", 10, 1.0)  # type: ignore[arg-type]
        print("[FAIL] Test 3: Should have raised TypeError, got:", bad)
    except TypeError:
        print("[PASS] Test 3: Wrong date type detected")

    # Test 4: execute_sql
    try:
        sql_count = rm.execute_sql("SELECT COUNT(*) FROM sale_records")
        assert sql_count[0][0] == 6  # 上面插入了 3 + 3 条
        print("[PASS] Test 4: execute_sql works")
    except Exception as e:
        print("[FAIL] Test 4 error:", e)

    # Test 5: SupplierPriceRecord
    try:
        sp1 = SupplierPriceRecord("SUP001", "SKU_A", date(2024, 1, 1), 3.5)
        sp2 = SupplierPriceRecord("SUP001", "SKU_A", date(2024, 1, 2), 3.8)
        sp3 = SupplierPriceRecord("SUP001", "SKU_B", date(2024, 1, 1), 2.0)

        rm.add_supplier_price(sp1)
        rm.add_supplier_price(sp2)
        rm.add_supplier_price(sp3)

        prices_a = rm.read_supplier_prices("SUP001", "SKU_A")
        assert len(prices_a) == 2
        assert prices_a[0].supplier_id == "SUP001"
        assert prices_a[0].sku_id == "SKU_A"

        prices_range = rm.read_supplier_prices(
            supplier_id="SUP001",
            sku_id="SKU_A",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
        )
        assert len(prices_range) == 1
        assert prices_range[0].price == 3.8

        print("[PASS] Test 5: SupplierPriceRecord works")
    except Exception as e:
        print("[FAIL] Test 5 error:", e)

    # Test 6: SupplierOrderRecord
    try:
        so1 = SupplierOrderRecord(
            supplier_id="SUP001",
            order_date=date(2024, 2, 1),
            arrival_date=date(2024, 2, 3),
            items={"SKU_A": 100, "SKU_B": 50},
        )
        so2 = SupplierOrderRecord(
            supplier_id="SUP001",
            order_date=date(2024, 2, 10),
            arrival_date=date(2024, 2, 12),
            items={"SKU_A": 30},
        )

        rm.add_supplier_order(so1)
        rm.add_supplier_order(so2)

        orders_all = rm.read_supplier_orders("SUP001")
        assert len(orders_all) == 2
        assert orders_all[0].shipping_days == 2

        orders_range = rm.read_supplier_orders(
            supplier_id="SUP001",
            start_order_date=date(2024, 2, 2),
            end_order_date=date(2024, 2, 11),
        )
        assert len(orders_range) == 1
        assert orders_range[0].order_date == date(2024, 2, 10)
        assert orders_range[0].items["SKU_A"] == 30

        print("[PASS] Test 6: SupplierOrderRecord works")
    except Exception as e:
        print("[FAIL] Test 6 error:", e)

    shutil.rmtree(temp_dir)
    print(f"[INFO] Cleaned temporary directory: {temp_dir}")
    print("===== Tests Finished =====")


if __name__ == "__main__":
    run_tests()
