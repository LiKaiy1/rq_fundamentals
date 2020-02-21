# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。

# 1、回测区间：回归(2014-01-01~2016-01-01)
#           回测2016-01-01  ~  2018-01-01
# 2、选股：
#   选股区间：沪深300
#   选股因子：经过因子分析之后的若干因子，可以不知方向
#   选股权重：回归训练的权重
#   数据处理：缺失值、去极值、标准化、市值中心化处理（防止选股集中）
# 3、调仓周期：
#   调仓：每月进行一次调仓
#   交易规则：卖出已持有的股票
#          买入新的股票池当中的股票

# 步骤
# 准备因子数据
# 处理数据函数
# dealwith_data(context)
# 选股函数(每月调整股票池)
# select_stocklist(context)
# 定期调仓函数
# rebalance(context)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 定义沪深300的股票列表
    context.hs300 = index_components("000300.XSHG")

    # 初始化股票因子权重
    context.weights = np.array(
        [-0.01864979, -0.04537212, -0.18487143, -0.06092573, 0.18599453, -0.02088234, 0.03341527, 0.91743347,
         -0.8066782])

    # 定义股票池数量
    context.stocknum = 20

    # 定义定时每月运行的函数
    scheduler.run_monthly(regression_select, tradingday=1)


def regression_select(context, bar_dict):
    """回归法预测选股逻辑
    """
    # 1、查询因子数据
    # 查询因子顺序跟建立回归系数顺序一样
    q = query(fundamentals.eod_derivative_indicator.pe_ratio,
              fundamentals.eod_derivative_indicator.pb_ratio,
              fundamentals.eod_derivative_indicator.market_cap,
              fundamentals.financial_indicator.ev,
              fundamentals.financial_indicator.return_on_asset_net_profit,
              fundamentals.financial_indicator.du_return_on_equity,
              fundamentals.financial_indicator.earnings_per_share,
              fundamentals.income_statement.revenue,
              fundamentals.income_statement.total_expense).filter(fundamentals.stockcode.in_(context.hs300))

    fund = get_fundamentals(q)

    # 行列进行转置
    context.factors_data = fund.T

    # 2、因子（特征值）数据进行处理
    dealwith_data(context)

    # 3、根据每月预测下月的收益率大小替换股票池
    select_stocklist(context)

    # 4、根据股票池的股票列表，进行调仓
    rebalance(context)


def dealwith_data(context):
    """
    contex: 包含因子数据
    需要做的处理：删除空值、去极值、标准化、因子的市值中性化
    """
    # 删除空值
    context.factors_data = context.factors_data.dropna()

    # 市值因子，去做特征值给其它因子中性化处理
    # 市值因子因子不进行去极值、标准化处理
    market_cap_factor = context.factors_data['market_cap']

    # 去极值标准化，循环对每个因子进行处理
    for name in context.factors_data.columns:

        # 对每个因子进行去极值、标准化处理
        context.factors_data[name] = mad(context.factors_data[name])
        context.factors_data[name] = stand(context.factors_data[name])

        # 对因子（除了martket_cap本身不需要中性化）中性化处理
        # 特征值：market_cap_factor
        # 目标： name每个因子
        if name == "market_cap":
            continue

        x = market_cap_factor
        y = context.factors_data[name]

        # 建立回归方程、市值中性化
        lr = LinearRegression()

        # x:要求二维，y：要求1维
        lr.fit(x.reshape(-1, 1), y)

        y_predict = lr.predict(x.reshape(-1, 1))

        # 得出误差进行替换原有因子值
        context.factors_data[name] = y - y_predict


def select_stocklist(context):
    """
    回归计算预测得出收益率结果，筛选收益率高的股票
    """

    # 特征值是：context.factors_data （300， 9）
    # 系数：因子权重
    # 进行矩阵运算，预测收益率
    # （m行，n列）* （n行，l列） = （m行， l列）
    # (300, 9) * (9, 1) = (300, 1)
    # logger.info(context.factors_data.shape)

    # 预测收益率，如果收益高，那么接下来的下一个月都持有收益高的
    # 这写股票
    stock_return = np.dot(context.factors_data.values, context.weights)

    # 赋值给因子数据,注意都是默认对应的股票代码和收益率
    context.factors_data['stock_return'] = stock_return

    # 进行收益率的排序
    # 修改：按照从大到小的收益率排序，选择前
    context.stock_list = context.factors_data.sort_values(by='stock_return', ascending=False).index[:context.stocknum]

    # logger.info(context.stock_list)


def rebalance(context):
    """进行调仓位的函数
    """
    # 卖出
    for stock in context.portfolio.positions.keys():

        if context.portfolio.positions[stock].quantity > 0:

            if stock not in context.stock_list:
                order_target_percent(stock, 0)

    # 买入
    for stock in context.stock_list:
        order_target_percent(stock, 1.0 / context.stocknum)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # TODO: 开始编写你的算法吧！
    # order_shares(context.s1, 1000)
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


def mad(factor):
    """3倍中位数去极值
    """
    # 求出因子值的中位数
    med = np.median(factor)

    # 求出因子值与中位数的差值，进行绝对值
    mad = np.median(np.abs(factor - med))

    # 定义几倍的中位数上下限
    high = med + (3 * 1.4826 * mad)
    low = med - (3 * 1.4826 * mad)

    # 替换上下限以外的值
    factor = np.where(factor > high, high, factor)
    factor = np.where(factor < low, low, factor)
    return factor


def stand(factor):
    """标准化
    """
    mean = np.mean(factor)
    std = np.std(factor)
    return (factor - mean) / std# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。

# 1、回测区间：回归(2014-01-01~2016-01-01)
#           回测2016-01-01  ~  2018-01-01
# 2、选股：
#   选股区间：沪深300
#   选股因子：经过因子分析之后的若干因子，可以不知方向
#   选股权重：回归训练的权重
#   数据处理：缺失值、去极值、标准化、市值中心化处理（防止选股集中）
# 3、调仓周期：
#   调仓：每月进行一次调仓
#   交易规则：卖出已持有的股票
#          买入新的股票池当中的股票

# 步骤
# 准备因子数据
# 处理数据函数
# dealwith_data(context)
# 选股函数(每月调整股票池)
# select_stocklist(context)
# 定期调仓函数
# rebalance(context)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 定义沪深300的股票列表
    context.hs300 = index_components("000300.XSHG")

    # 初始化股票因子权重
    context.weights = np.array([-0.01864979, -0.04537212, -0.18487143, -0.06092573,  0.18599453,-0.02088234,  0.03341527,  0.91743347, -0.8066782])

    # 定义股票池数量
    context.stocknum = 20

    # 定义定时每月运行的函数
    scheduler.run_monthly(regression_select, tradingday=1)


def regression_select(context, bar_dict):
    """回归法预测选股逻辑
    """
    # 1、查询因子数据
    # 查询因子顺序跟建立回归系数顺序一样
    q = query(fundamentals.eod_derivative_indicator.pe_ratio,
            fundamentals.eod_derivative_indicator.pb_ratio,
            fundamentals.eod_derivative_indicator.market_cap,
            fundamentals.financial_indicator.ev,
            fundamentals.financial_indicator.return_on_asset_net_profit,
            fundamentals.financial_indicator.du_return_on_equity,
            fundamentals.financial_indicator.earnings_per_share,
            fundamentals.income_statement.revenue,
            fundamentals.income_statement.total_expense).filter(fundamentals.stockcode.in_(context.hs300))

    fund = get_fundamentals(q)

    # 行列进行转置
    context.factors_data = fund.T

    # 2、因子（特征值）数据进行处理
    dealwith_data(context)

    # 3、根据每月预测下月的收益率大小替换股票池
    select_stocklist(context)

    # 4、根据股票池的股票列表，进行调仓
    rebalance(context)


def dealwith_data(context):
    """
    contex: 包含因子数据
    需要做的处理：删除空值、去极值、标准化、因子的市值中性化
    """
    # 删除空值
    context.factors_data = context.factors_data.dropna()

    # 市值因子，去做特征值给其它因子中性化处理
    # 市值因子因子不进行去极值、标准化处理
    market_cap_factor = context.factors_data['market_cap']

    # 去极值标准化，循环对每个因子进行处理
    for name in context.factors_data.columns:

        # 对每个因子进行去极值、标准化处理
        context.factors_data[name] = mad(context.factors_data[name])
        context.factors_data[name] = stand(context.factors_data[name])

        # 对因子（除了martket_cap本身不需要中性化）中性化处理
        # 特征值：market_cap_factor
        # 目标： name每个因子
        if name == "market_cap":
            continue

        x = market_cap_factor
        y = context.factors_data[name]

        # 建立回归方程、市值中性化
        lr = LinearRegression()

        # x:要求二维，y：要求1维
        lr.fit(x.reshape(-1, 1), y)

        y_predict = lr.predict(x.reshape(-1, 1))

        # 得出误差进行替换原有因子值
        context.factors_data[name] = y - y_predict


def select_stocklist(context):
    """
    回归计算预测得出收益率结果，筛选收益率高的股票
    """

    # 特征值是：context.factors_data （300， 9）
    # 系数：因子权重
    # 进行矩阵运算，预测收益率
    # （m行，n列）* （n行，l列） = （m行， l列）
    # (300, 9) * (9, 1) = (300, 1)
    # logger.info(context.factors_data.shape)

    # 预测收益率，如果收益高，那么接下来的下一个月都持有收益高的
    # 这写股票
    stock_return = np.dot(context.factors_data.values, context.weights)

    # 赋值给因子数据,注意都是默认对应的股票代码和收益率
    context.factors_data['stock_return'] = stock_return

    # 进行收益率的排序
    # 修改：按照从大到小的收益率排序，选择前
    context.stock_list = context.factors_data.sort_values(by='stock_return', ascending=False).index[:context.stocknum]

    # logger.info(context.stock_list)

def rebalance(context):
    """进行调仓位的函数
    """
    # 卖出
    for stock in context.portfolio.positions.keys():

        if context.portfolio.positions[stock].quantity > 0:

            if stock not in context.stock_list:

                order_target_percent(stock, 0)

    # 买入
    for stock in context.stock_list:

        order_target_percent(stock, 1.0 / context.stocknum)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # TODO: 开始编写你的算法吧！
    # order_shares(context.s1, 1000)
    pass

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


def mad(factor):
    """3倍中位数去极值
    """
    # 求出因子值的中位数
    med = np.median(factor)

    # 求出因子值与中位数的差值，进行绝对值
    mad = np.median(np.abs(factor - med))

    # 定义几倍的中位数上下限
    high = med + (3 * 1.4826 * mad)
    low = med - (3 * 1.4826 * mad)

    # 替换上下限以外的值
    factor = np.where(factor > high, high, factor)
    factor = np.where(factor < low, low, factor)
    return factor

def stand(factor):
    """标准化
    """
    mean = np.mean(factor)
    std = np.std(factor)
    return (factor - mean)/std