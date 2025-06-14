from flask import Flask, send_file, request, render_template, session, jsonify, abort
from flask import send_from_directory
from PIL import Image, ImageFilter
from PIL import UnidentifiedImageError as PILUnidentifiedImageError
from functools import lru_cache
from werkzeug.utils import secure_filename
from pathlib import Path
import io
import os
import json
import numpy as np
import time
import csv
import logging
import gc

# Debug tools
# import logging
import pdb  # pdb.set_trace()

MAX_PIXELS = 50 * 1024 * 1024

TimeMMs = False
Footstep = False

#app = Flask(__name__, template_folder="templates", static_folder="static")
app = Flask(__name__)

app.secret_key = "my_development_secret_123456"  # session を使う

# pre_processのためのキャッシュ
preprocess_cache = {}

# ロギングの設定（ログレベルをDEBUGに設定）
# logging.basicConfig(level=logging.DEBUG)

# class definition #############################################################

class ConfigError(Exception):
    """設定に関する基底例外クラス"""

class MissingConfigKeyError(ConfigError):
    """設定ファイルに必要なキーが見つからない"""

class ImageFileNotFoundError(ConfigError):
    """指定された画像ファイルが存在しない"""

class OutOfRangeValueError(ConfigError):
    """指定された値が範囲外である"""

class EvenValueExpectedError(ConfigError):
    """偶数が期待される値に奇数が指定された"""

class ImageUnidentifiedError(ConfigError):
    """画像の識別に失敗した"""

class MemoryError(ConfigError):
    """メモリーがオーバーフローした"""

class ModeParameter:
    def __init__(self, view_mode, hosei_mode, upend, mirror_mode, projection):
        # モード/フラグ/パラメーター
        self.view_mode = view_mode  # ビューモード
        self.hosei_mode = hosei_mode  # 補正モード
        self.upend = upend  # 直立/倒立-フラグ
        self.projection = projection  # 光学系射影方式
        self.mirror_mode = mirror_mode  # 鏡面反転

    def getval(self):
        return [self.view_mode, self.hosei_mode, self.upend, self.mirror_mode, self.projection]


class FisheyeLens:
    def __init__(self, pm, dd):
        # 魚眼レンズ
        self.pm = pm  # 射映方式 [-]
        self.pm = dd  # dummy data

    def getval(self):
        return [self.pm, self.dd]


class HyperboloidMirror:
    def __init__(self, cc, bb):
        # 双曲面ミラー
        self.cc = cc  # 双曲面係数c [-]
        self.bb = bb  # 双曲面係数b [-]

    def getval(self):
        return [self.cc, self.bb]


class SphereMirror:
    def __init__(self, rr_u, hh_u):
        # 球面ミラー
        self.rr_u = rr_u  # 球半径 [mm]
        self.hh_u = hh_u  # 球中心高さ [mm]

    def getval(self):
        return [self.rr_u, self.hh_u]


class ImagingRange:
    def __init__(self, limsup, liminf):
        # 全射影方式の共通クラス
        self.limsup = limsup  # 0上限角 [deg] Bmaxから改名
        self.liminf = liminf  # 1下限角 [deg] Amaxから改名

    def getval(self):
        return [self.limsup, self.liminf]


class ViewpointAngles:
    def __init__(self, ang_x, ang_y, ang_z):
        # 視点を決める角度
        self.ang_x = ang_x  # x軸周り角度 Ψ [rad]
        self.ang_y = ang_y  # y軸周り角度 Θ [rad](=0.0;未使用)
        self.ang_z = ang_z  # z軸周り角度 Φ [rad]

    def getval(self):
        return [self.ang_x, self.ang_y, self.ang_z]


class ScreenSize:
    def __init__(self, twidth1, theight1, dhagv, twidth2, theight2):
        # 画面サイズ(表示ウィンドウサイズ)
        self.twidth1 = twidth1  # 画像幅(軽量ハイスピード) [px]
        self.theight1 = theight1  # 画像高さ(軽量ハイスピード) [px]
        self.dhagv = dhagv  # 水平画角 [deg]
        self.twidth2 = twidth2  # 画像幅(ハイレゾリューション) [px]
        self.theight2 = theight2  # 画像高さ(ハイレゾリューション) [px]

    def getval(self):
        return [
            self.twidth1,
            self.theight1,
            self.dhagv,
            self.twidth2,
            self.theight2
        ]


class CurvRad:
    def __init__(self, rv, fv):
        # 画面の曲率半径と係数 fv(計算値)
        self.rv = rv  # 仮想曲率半径 [px]
        self.fv = fv  # 係数 fv [-]

    def getval(self):
        return [self.rv, self.fv]


class ImageSource:
    def __init__(self, simg, dd_u):
        # 元画像のファイル名と直径
        self.simg = simg  # 画像ファイル名(拡張子を含む)
        self.dd_u = dd_u  # 全方位画像の直径画素数 [px]

    def getval(self):
        return [self.simg, self.dd_u]


class MaskColor:
    def __init__(self, fore_color, back_color):
        # マスクするカラー値(オプション)
        self.fore_color = fore_color  # 前景カラー値(補正画像内)
        self.back_color = back_color  # 後景カラー値(補正画像外)

    def getval(self):
        return [self.fore_color, self.back_color]

class FreeText:
    def __init__(self, str0, str1, str2, str3):
        # モード/フラグ/パラメーター
        self.str0 = str0  # ストリング0
        self.str1 = str1  # ストリング1
        self.str2 = str2  # ストリング2
        self.str3 = str3  # ストリング3

    def getval(self):
        return [self.str0, self.str1, self.str2, self.str3]
# end of class definition ######################################################


########################################################################
#  "templates" folder にある son-file の設定データを読み込む
########################################################################
def get_setting(file_path):
    
    with open(file_path, encoding='utf-8') as file:
        setting = json.load(file)

    dic = setting[0]

#-- ModeParameter ------------------------------------------------------
    if 'view_mode' in dic:
        view_mode = dic['view_mode']
    else:
        raise MissingConfigKeyError("The Config Key (view_mode) is missing.")
    if Footstep:
        print('----- after view_mode -----', flush = True)

    if 'hosei_mode' in dic:
        hosei_mode = dic['hosei_mode']
    else:
        raise MissingConfigKeyError("The Config Key (hosei_mode) is missing.")
    if Footstep:
        print('----- after hosei_mode -----', flush = True)

    if 'upright' in dic:
        upright = dic['upright']
    else:
        raise MissingConfigKeyError("The Config Key (upright) is missing.")
    if Footstep:
        print('----- after upright -----', flush = True)

    if 'projection' in dic:
        projection = dic['projection']
    else:
        raise MissingConfigKeyError("The Config Key (projection) is missing.")
    if Footstep:
        print('----- after projection -----', flush = True)

    # 実像/虚像(鏡像)の判定
    mirror_mode = False
    if projection == 4:
        mirror_mode = True
    elif projection == 5:
        mirror_mode = True
    elif projection == 6:
        mirror_mode = True
    gmp = ModeParameter(
        view_mode, hosei_mode, upright, mirror_mode, projection
    )

#-- FisheyeLens 0,1,2,3 ------------------------------------------------
#-- HyperboloidMirror 4 ------------------------------------------------
#-- SphereMirror 5,6 ---------------------------------------------------
    prj = projection
    if prj == 0:  # 等距離射影(魚眼レンズ,放物面ミラー,PAL)
        pm = 0
        dd = 0
        gops = FisheyeLens(pm, dd)
    elif prj == 1:  # 立体射影/平射影(魚眼レンズ)
        pm = 1
        dd = 0
        gops = FisheyeLens(pm, dd)
    elif prj == 2:  # 等立体角射影(魚眼レンズ)
        pm = 2
        dd = 0
        gops = FisheyeLens(pm, dd)
    elif prj == 3:  # 正射影(魚眼レンズ)
        pm = 3
        dd = 0
        gops = FisheyeLens(pm, dd)
    elif prj == 4:  # 双曲面ミラー＋接写レンズ
        if 'cc' in dic:
            cc = dic['cc']
        else:
            raise MissingConfigKeyError("The Config Key (cc) is missing.")
        if 'bb' in dic:
            bb = dic['bb']
        else:
            raise MissingConfigKeyError("The Config Key (bb) is missing.")
        gops = HyperboloidMirror(cc, bb)
    elif prj == 5 or prj == 6:  # 半球面ミラー＋接写レンズ
        # 5:設計最大入射角基準
        # 6:理論最大入射角基準
        if 'rr_u' in dic:
            rr_u = dic['rr_u']
        else:
            raise MissingConfigKeyError("The Config Key (rr_u) is missing.")
        if 'hh_u' in dic:
            hh_u = dic['hh_u']
        else:
            raise MissingConfigKeyError("The Config Key (hh_u) is missing.")
        gops = SphereMirror(rr_u, hh_u)

#-- ImagingRange -------------------------------------------------------
    if 'limsup' in dic:
        limsup = dic['limsup']
    else:
        raise MissingConfigKeyError("The Config Key (limsup) is missing.")
    if 'liminf' in dic:
        liminf = dic['liminf']
    else:
        raise MissingConfigKeyError("The Config Key (liminf) is missing.")
    gir = ImagingRange(limsup, liminf)

#-- ViewpointAngles ----------------------------------------------------
    if 'ang_x' in dic:
        ang_x = dic['ang_x']
    else:
        raise MissingConfigKeyError("The Config Key (ang_x) is missing.")
    if 'ang_y' in dic:
        ang_y = dic['ang_y']
    else:
        ang_y = 0
    if 'ang_z' in dic:
        ang_z = dic['ang_z']
    else:
        raise MissingConfigKeyError("The Config Key (ang_z) is missing.")
    gva = ViewpointAngles(ang_x, ang_y, ang_z)

#-- ScreenSize ---------------------------------------------------------
    if 'twidth' in dic:
        twidth2 = dic['twidth']  # 画像幅(ハイレゾリューション) [px]
        twidth1 = round(twidth2/(2.)/2.)*2  # 画像幅(軽量ハイスピード) [px]c
    else:
        raise MissingConfigKeyError("The Config Key (twidth) is missing.")
    if twidth2 % 2 != 0:
        raise EvenValueExpectedError(
            f"(twidth) must be an even number（Config Value: {twidth2}）"
        )

    if 'theight' in dic:
        theight2 = dic['theight']  # 画像高さ(ハイレゾリューション) [px]
        theight1 = round(theight2*twidth1/twidth2)  # 画像高さ(軽量ハイスピード) [px]
    else:
        raise MissingConfigKeyError("The Config Key (theight) is missing.")
    if 'dhagv' in dic:
        dhagv = dic['dhagv']  # 水平画角 [deg]
    else:
        raise MissingConfigKeyError("The Config Key (dhagv) is missing.")

    gsz = ScreenSize(twidth1, theight1, dhagv, twidth2, theight2)

#-- ImageSource --------------------------------------------------------
    if 'simg' in dic:
        simg = dic['simg']
    else:
        raise MissingConfigKeyError("The Config Key (simg) is missing.")
    if 'dd_u' in dic:
        # optional data
        dd_u = dic['dd_u']
    else:
        dd_u = 0
    gim = ImageSource(simg, dd_u)

#--　MaskColor　--------------------------------------------------------
    fore_color = (96, 96, 96)  # default foreground color
    if 'fore_color' in dic:
        fore_color = tuple(dic['fore_color'])
    back_color = (0, 0, 0)  # default background color
    if 'back_color' in dic:
        back_color = tuple(dic['back_color'])
    gmc = MaskColor(fore_color, back_color)

#--　FreeText　---------------------------------------------------------
    str0 = ''  # ストリング0
    if 'str0' in dic:
        str0 = dic['str0']
    str1 = ''  # ストリング1
    if 'str1' in dic:
        str1 = dic['str1']
    str2 = ''  # ストリング2
    if 'str2' in dic:
        str2 = dic['str2']
    str3 = ''  # ストリング3
    if 'str3' in dic:
        str3 = dic['str3']
    gft = FreeText(str0, str1, str2, str3)

    max_agv = gir.getval()[0]*2 if gmp.getval()[0] == 6 else 360
    if dhagv > max_agv:
        raise OutOfRangeValueError("Horizontal angle of view is out of range")

    return gmp, gops, gir, gva, gsz, gim, gmc, gft
# End of get_setting ()

@lru_cache(maxsize=32768)
def squaring(n):
    return n*n

########################################################################
#  球の無次元高さh(hh)と角度Θ(a_theta_u[rad])からβ[rad]を求める
#  ファンクション[tan_beta]はtanβを求めていた旧仕様の名残り
########################################################################
def tan_beta(hh, a_theta_u):
    if hh == 0.0:
        return 0.0
    # βの最大値
    a_beta = np.arccos(1.0/hh)
    # Θの最大値
    work = a_beta + np.pi/2.0
    # βの第一近似解
    a_beta *= (a_theta_u / work)
    # βの第二近似解
    work = 2.0*a_beta+np.arctan2(np.sin(a_beta), hh-np.cos(a_beta))
    a_beta *= (a_theta_u / work)
    for i in range(2):
        work = np.cos(a_beta)
        fc_beta = 2.0*a_beta + np.arctan2(np.sin(a_beta), hh-work)
        fc_beta -= a_theta_u
        fd_beta = (1.0 + (-3.0*work + 2.0*hh)*hh)
        fd_beta /= (1.0 + (-2.0*work + hh)*hh)
        a_beta -= (fc_beta/fd_beta)
    return a_beta
# End of tan_beta ()


########################################################################
#  最大像直径(dd_u[px])から方向余弦 c3のテーブル(faxt[][])を求める
########################################################################
def set_fast(fast, dd_u, projection, gops, gir):
    limsup = gir.getval()[0]
    if projection == 0:  # 等距離射影(魚眼レンズ,放物面ミラー,PAL)
        # mirror_mode = False
        r_ref = round((dd_u * 90.0/limsup) / 2.0)  # 基準像高(90deg)
        ff = 2.0 * r_ref / np.pi  # 光学系の焦点距離に相当
        dc3 = 1.0 / 1000
        if limsup <= 90:
            for i in range(1000):
                c3 = i * dc3
                fast[0][i] = (np.arccos(+c3) * ff) / \
                            (np.sqrt(1.0 - np.power(c3, 2)))
        else:
            for i in range(1000):
                c3 = i * dc3
                fast[0][i] = (np.arccos(+c3) * ff) / \
                            (np.sqrt(1.0 - np.power(c3, 2)))
                fast[1][i] = (np.arccos(-c3) * ff) / \
                            (np.sqrt(1.0 - np.power(c3, 2)))
    elif projection == 1:  # 立体射影/平射影(魚眼レンズ)
        # mirror_mode = False
        r_ref = round((dd_u * 1.0/np.tan(np.radians(limsup/2.0))) / 2.0)
        dc3 = 1.0 / 1000
        if limsup <= 90:
            for i in range(1000):
                c3 = i * dc3
                fast[0][i] = r_ref / (1.0 + c3)
        else:
            for i in range(1000):
                c3 = i * dc3
                fast[0][i] = r_ref / (1.0 + c3)
                fast[1][i] = r_ref / (1.0 - c3)
    elif projection == 2:  # 等立体角射影(魚眼レンズ)
        # mirror_mode = False
        r_ref = round((dd_u * np.sqrt(2.0)/np.sin(np.radians(limsup/2.0))) / 4.0)
        dc3 = 1.0 / 1000
        if limsup <= 90:
            for i in range(1000):
                c3 = i * dc3
                fast[0][i] = r_ref / np.sqrt(1.0 + c3)
        else:
            for i in range(1000):
                c3 = i * dc3
                fast[0][i] = r_ref / (1.0 + c3)
                fast[1][i] = r_ref / (1.0 - c3)
    elif projection == 3:  # 正射影(魚眼レンズ)
        # mirror_mode = False
        r_ref = round((dd_u * 1.0/np.sin(np.radians(limsup))) / 2.0)
        # 0 ≤ Θ ≤ 90 [deg]
        dc3 = 1.0 / 1000
        for i in range(1000):
            fast[0][i] = r_ref
    elif projection == 4:  # 双曲面ミラー＋接写レンズ
        # mirror_mode = True
        rr = round(dd_u/2.0)  # 最大像高(limsupに対する)
        cos1 = np.cos(np.radians(limsup))  # 最大画角に対する余弦
        sin1 = np.sin(np.radians(limsup))  # 最大画角に対する正弦
        cc = gops.getval()[0]
        bb = gops.getval()[1]
        ccq = np.power(cc, 2)
        bbq = np.power(bb, 2)
        e = (2.0*bb*cc)/(bbq+ccq)
        r_ref = round(rr*(e+cos1)/(e*sin1))
        dc3 = 1.0 / 1000
        if limsup <= 90:
            for i in range(1000):
                c3 = i * dc3
                fast[0][i] = (r_ref*e) / (e + c3)
        else:
            for i in range(1000):  # (ccq+bbq)*fh
                c3 = i * dc3
                fast[0][i] = (r_ref*e) / (e + c3)
                fast[1][i] = (r_ref*e) / (e - c3)
    elif projection == 5:  # 半球面ミラー＋接写レンズ(設計最大入射角基準)
        # mirror_mode = True
        r_ref = round(dd_u/2.0)  # 設計最大入射角に対応する半径[px]
        rr_u = gops.getval()[0]  # 球半径[mm]
        hh_u = gops.getval()[1]  # 球中心のレンズ主点からの高さ[mm]
        hh = hh_u/rr_u  # 無次元高さ[-]
        a_theta_u = np.radians(limsup)
        a_beta = tan_beta(hh, a_theta_u)
        work = (r_ref*(hh - np.cos(a_beta)))/np.sin(a_beta)
        dc3 = 1.0 / 1000
        if limsup <= 90:
            for i in range(1000):
                c3 = i * dc3
                a_theta_u = np.arccos(+c3)
                a_beta = tan_beta(hh, a_theta_u)
                fast[0][i] = (work*np.sin(a_beta)) / \
                            (np.sin(a_theta_u)*(hh-np.cos(a_beta)))
        else:
            for i in range(1000):
                c3 = i * dc3
                a_theta_u = np.arccos(+c3)
                a_beta = tan_beta(hh, a_theta_u)
                fast[0][i] = (work*np.sin(a_beta)) / \
                            (np.sin(a_theta_u)*(hh-np.cos(a_beta)))
                a_theta_u = np.arccos(-c3)
                a_beta = tan_beta(hh, a_theta_u)
                fast[1][i] = (work*np.sin(a_beta)) / \
                            (np.sin(a_theta_u)*(hh-np.cos(a_beta)))
    elif projection == 6:  # 半球面ミラー＋接写レンズ(理論最大入射角基準)
        # mirror_mode = True
        r_ref = round(dd_u/2.0)  # 基準像高(全方位画像に対する最大像高)
        rr_u = gops.getval()[0]
        hh_u = gops.getval()[1]
        hh = hh_u/rr_u  # 球面中心のレンズ主点からの無次元高さ
        work = r_ref*np.sqrt(np.power(hh, 2) - 1.0)
        dc3 = 1.0 / 1000
        if limsup <= 90:
            for i in range(1000):
                c3 = i * dc3
                a_theta_u = np.arccos(+c3)
                a_beta = tan_beta(hh, a_theta_u)
                fast[0][i] = (work*np.sin(a_beta)) / \
                            (np.sin(a_theta_u)*(hh-np.cos(a_beta)))
        else:
            for i in range(1000):
                c3 = i * dc3
                a_theta_u = np.arccos(+c3)
                a_beta = tan_beta(hh, a_theta_u)
                fast[0][i] = (work*np.sin(a_beta)) / \
                            (np.sin(a_theta_u)*(hh-np.cos(a_beta)))
                a_theta_u = np.arccos(-c3)
                a_beta = tan_beta(hh, a_theta_u)
                fast[1][i] = (work*np.sin(a_beta)) / \
                            (np.sin(a_theta_u)*(hh-np.cos(a_beta)))
    else:  # 半球面ミラー(旧コード)
        # mirror_mode = True
        rr = round(dd_u/2.0)  # 基準像高(limsupに対する最大像高)
        rr_u = gops.getval()[0]
        hh_u = gops.getval()[1]
        hh = hh_u/rr_u
        a_theta_u = np.radians(limsup)
        rep_beta = 1.0/tan_beta(hh, a_theta_u)
        dc3 = 1.0 / 1000
        if limsup <= 90:
            for i in range(1000):
                c3 = i * dc3
                a_theta_u = np.arccos(+c3)
                fast[0][i] = (rr*tan_beta(hh, a_theta_u)*rep_beta) / \
                            (np.sin(a_theta_u))
        else:
            for i in range(1000):
                c3 = i * dc3
                a_theta_u = np.arccos(+c3)
                fast[0][i] = (rr*tan_beta(hh, a_theta_u)*rep_beta) / \
                            (np.sin(a_theta_u))
                a_theta_u = np.arccos(-c3)
                fast[1][i] = (rr*tan_beta(hh, a_theta_u)*rep_beta) / \
                            (np.sin(a_theta_u))
# End of set_fast ()
#
# リングビューモード対応
# 方向余弦を設定または
# 縦スクロール時に方向余弦を再設定
#
def load_tcp_rv(tcp, rv, fv, twidth0, theight0, nstep, gbias, gmp):  # リングビュー 3-(0/1)-(0/1)
    # global _gbias
    # if nstep >= 3:
    #     # HSモードだけの実行･非実行判断
    #     if gbias == _gbias:
    #         # biasに変更なければ実行不要
    #         return
    # current bias を保存
    # _gbias = gbias
    hosei_mode = gmp.getval()[1]  # 補正モード
    upright = gmp.getval()[2]  # 直立/倒立フラグ
    # twidth0 = gsz.getval()[0]  # スクリーン幅[pxl]
    # theight0 = gsz.getval()[1]  # スクリーン高[pxl]
    hgh = (twidth0 - 1.0) / 2.0
    if hosei_mode == 0:  # 球面補正(球面射影) 3-0-(0/1)
        if upright:  # 直立 3-0-0
            # rint('Load_tcp_rv/リングビュー/球面補正(球面射影)/直立 3-0-0')
            # 昇順↓ ######## 未軽量化
            for ivp in range(theight0):
                yp = ivp - (theight0-1.0)/2.0 - gbias
                iva = ivp 
                # 昇順→(1)
                for ihp in range(0, twidth0-1+nstep, nstep):
                    if ihp > twidth0 - 1:
                        ihp = twidth0 - 1
                    xp = ihp - hgh
                    ah = np.sqrt(xp*xp + yp*yp) / rv
                    cnh = np.cos(ah)
                    snh = np.sin(ah)
                    av = np.arctan2(yp, xp)
                    cnv = np.cos(av)
                    snv = np.sin(av)
                    if fv != 0:
                        ah = np.arctan2(snh, (cnh - fv))
                        cnh = np.cos(ah)
                        snh = np.sin(ah)
                    iha = ihp
                    i = iha + twidth0*iva
                    tcp[i][0] = snh*cnv
                    tcp[i][1] = snh*snv
                    tcp[i][2] = cnh
        # Break of 直立 3-0-0
        else:  # 倒立 3-0-1
            # rint('Load_tcp_rv/リングビュー/球面補正(球面射影)/倒立 3-0-1')
            # 降順↑ ######## 未軽量化
            for ivp in reversed(range(theight0)):
                yp = ivp - (theight0-1.0)/2.0 - gbias  # - 106 (bias new)
                iva = theight0-1-ivp  # new
                # 降順←(2)
                for ihp in range(twidth0-1, -nstep, -nstep):
                    if ihp < 0:
                        ihp = 0
                    xp = ihp - hgh
                    c1 = np.sqrt(xp*xp + yp*yp) / rv
                    c2 = np.arctan2(yp, xp)
                    cos1 = np.cos(c1)
                    sin1 = np.sin(c1)
                    cos2 = np.cos(c2)
                    sin2 = np.sin(c2)
                    if fv != 0:
                        c1 = np.arctan2(sin1, (cos1 - fv))
                        cos1 = np.cos(c1)
                        sin1 = np.sin(c1)
                    iha = twidth0-1-ihp  # new
                    i = iha + twidth0*iva
                    tcp[i][0] = sin1*cos2
                    tcp[i][1] = sin1*sin2
                    tcp[i][2] = cos1
        # Break of 倒立 3-0-1
    # Break of 球面補正(球面射影) 3-0-(0/1)
    elif hosei_mode == 1:  # 円筒面補正(円筒面射影) 3-1-(0/1)
        if upright:  # 直立 3-1-0
            # rint('Load_tcp_rv/リングビュー/円筒面補正(円筒面射影)/直立 3-1-0')
            rrv = 1.0 / rv
            # 昇順↓
            for ivp in range(theight0):
                yp = ivp - (theight0 - 1.0) / 2.0 - gbias
                iva = ivp
                # 昇順→(3)
                for ihp in range(0, twidth0-1+nstep, nstep):
                    if ihp > twidth0 - 1:
                        ihp = twidth0 - 1
                    hp = ihp - hgh
                    ah = hp * rrv
                    xp = rv * np.sin(ah)
                    zp = rv * (np.cos(ah) - fv)
                    rrp = 1.0 / np.sqrt(xp*xp + yp*yp + zp*zp)
                    iha = ihp
                    i = iha + twidth0*iva
                    tcp[i][0] = xp * rrp
                    tcp[i][1] = yp * rrp
                    tcp[i][2] = zp * rrp
            # i = 0
            # for iyp in range(theight0):
            #     yp = iyp - (theight0 - 1.0) / 2.0 - gbias
            #     av = np.arctan2(yp, rv)
            #     cnv = np.cos(av)
            #     snv = np.sin(av)
            #     # 昇順→(3)
            #     for ixp in range(0, twidth0-1+nstep, nstep):
            #         if ixp > twidth0 - 1:
            #             ixp = twidth0 - 1
            #         xp = ixp - hgh
            #         ah = xp / rv
            #         cnh = np.cos(ah)
            #         snh = np.sin(ah)
            #         if fv != 0:
            #             sin2 = np.abs(snh)
            #             ah = np.arctan2(snh, (cnh-fv))
            #             cnh = np.cos(ah)
            #             snh= np.sin(ah)
            #             av = np.arctan2(yp*np.abs(snh), rv*sin2)
            #             cnv = np.cos(av)
            #             snv = np.sin(av)
            #         tcp[i][0] = snh*cnv
            #         tcp[i][1] = snv
            #         tcp[i][2] = cnh*cnv
            #         i += 1
        # Break of 直立 3-1-0
        else:  # 倒立 3-1-1
            # rint('Load_tcp_rv/リングビュー/円筒面補正(円筒面射影)/倒立 3-1-1')
            rrv = 1.0 / rv
            # 降順↑
            for ivp in reversed(range(theight0)):
                yp = ivp - (theight0 - 1.0) / 2.0 - gbias
                iva = theight0-1-ivp
                # 降順←(4)
                for ihp in range(twidth0-1, -nstep, -nstep):
                    if ihp < 0:
                        ihp = 0
                    hp = ihp - hgh
                    ah = hp * rrv
                    xp = rv * np.sin(ah)
                    zp = rv * (np.cos(ah) - fv)
                    rrp = 1.0 / np.sqrt(xp*xp + yp*yp + zp*zp)
                    iha = twidth0-1-ihp
                    i = iha + twidth0*iva
                    tcp[i][0] = xp * rrp
                    tcp[i][1] = yp * rrp
                    tcp[i][2] = zp * rrp
            # i = 0
            # for iyp in reversed(range(theight0)):
            #     yp = iyp - (theight0-1.0)/2.0 - gbias
            #     av = np.arctan2(yp, rv)
            #     cnv = np.cos(av)
            #     snv = np.sin(av)
            #     # 降順←(4)
            #     for ixp in range(twidth0-1, -nstep, -nstep):
            #         if ixp < 0:
            #             ixp = 0
            #         xp = ixp - hgh
            #         ah = xp / rv
            #         cnh = np.cos(ah)
            #         snh = np.sin(ah)
            #         if fv != 0:
            #             absinh = np.abs(snh)
            #             ah = np.arctan2(snh, (cnh-fv))
            #             cnh = np.cos(ah)
            #             snh = np.sin(ah)
            #             av = np.arctan2(yp*np.abs(snh), rv*absinh)
            #             cnv = np.cos(av)
            #             snv = np.sin(av)
            #         tcp[i][0] = snh*cnv
            #         tcp[i][1] = snv
            #         tcp[i][2] = cnh*cnv
            #         i += 1
        # Break of 倒立 3-1-1
    # Break of 円筒面補正 3-1-(0/1)
# End of load_tcp_rv () リングビュー 3


#
# センタービューモード対応
# 方向余弦を設定または
# 縦スクロール時に方向余弦を再設定
#
def load_tcp_cv(tcp, rv, fv, twidth0, theight0, nstep, gbias, gmp):  # センタービュー 6-(0/1)-(0/1)
    # global _gbias
    # if nstep >= 3:
    #     # HSモードだけの実行･非実行判断
    #     if gbias == _gbias:
    #         # biasに変更なければ実行不要
    #         return
    # current bias を保存
    # _gbias = gbias
    hosei_mode = gmp.getval()[1]  # 補正モード
    upright = gmp.getval()[2]  # 直立/倒立フラグ
    # twidth0 = gsz.getval()[0]  # スクリーン幅[pxl]
    # theight0 = gsz.getval()[1]  # スクリーン高[pxl]
    hgh = (twidth0 - 1.0) / 2.0
    if hosei_mode == 0:  # 球面補正(球面射影) 6-0-(0/1)
        # if gbias == 0 and nstep > 2:
            # return
        if upright:  # 直立 6-0-0
            # rint('load_tcp_cv/センタービュー/球面補正(球面射影)/直立 6-0-0')
            if False:
                rrv = 1.0 / rv
                # 昇順↓
                for ivp in range(theight0):
                    vp = ivp - (theight0-1.0)/2.0 - gbias
                    av = vp * rrv
                    yp = rv * np.sin(av)
                    iva = ivp
                    # 昇順→(5)
                    for ihp in range(0, twidth0-1+nstep, nstep):
                        if ihp > twidth0 - 1:
                            ihp = twidth0 - 1
                        hp = ihp - hgh
                        ah = hp * rrv
                        xp = rv * np.sin(ah)
                        rp = np.sqrt(hp*hp + vp*vp)
                        zp = rv * (np.cos(rp*rrv) - fv)
                        rrp = 1.0 / np.sqrt(xp*xp + yp*yp + zp*zp)
                        iha = ihp
                        i = iha + twidth0*iva
                        tcp[i][0] = xp * rrp
                        tcp[i][1] = yp * rrp
                        tcp[i][2] = zp * rrp
            else:
                for ivp in range(theight0):
                    yp = ivp - (theight0-1.0)/2.0 - gbias
                    iva = ivp
                    # 昇順→(5)
                    for ihp in range(0, twidth0-1+nstep, nstep):
                        if ihp > twidth0 - 1:
                            ihp = twidth0 - 1
                        xp = ihp - hgh
                        c1 = np.sqrt(xp*xp + yp*yp) / rv
                        c2 = np.arctan2(yp, xp)
                        cos1 = np.cos(c1)
                        sin1 = np.sin(c1)
                        cos2 = np.cos(c2)
                        sin2 = np.sin(c2)
                        if fv != 0:
                            c1 = np.arctan2(sin1, (cos1 - fv))
                            cos1 = np.cos(c1)
                            sin1 = np.sin(c1)
                        iha = ihp
                        i = iha + twidth0*iva
                        tcp[i][0] = sin1*cos2
                        tcp[i][1] = sin1*sin2
                        tcp[i][2] = cos1
        # Break of 直立 6-0-0
        else:  # 倒立 6-0-1
            # rint('load_tcp_cv/センタービュー/球面補正(球面射影)/倒立 6-0-1')
            if False:
                rrv = 1.0 / rv
                i = 0
                # 降順↑
                for ivp in reversed(range(theight0)):
                    vp = ivp - (theight0-1.0)/2.0 - gbias
                    av = vp * rrv
                    yp = rv * np.sin(av)
                    iva = theight0-1-ivp
                    # 降順←(6)
                    for ihp in range(twidth0-1, -nstep, -nstep):
                        if ihp < 0:
                            ihp = 0
                        hp = ihp - hgh
                        ah = hp * rrv
                        xp = rv * np.sin(ah)
                        rp = np.sqrt(hp*hp + vp*vp)
                        zp = rv * (np.cos(rp*rrv) - fv)
                        rrp = 1.0 / np.sqrt(xp*xp + yp*yp + zp*zp)
                        iha = twidht-1-ihp
                        i = iha + twidth0*iva
                        tcp[i][0] = xp * rrp
                        tcp[i][1] = yp * rrp
                        tcp[i][2] = zp * rrp
                        # pdb.set_trace()  # 倒立 6-0-1 debugged
                        i += 1
            else:
                for ivp in reversed(range(theight0)):
                    yp = ivp - (theight0-1.0)/2.0 - gbias
                    iva = theight0-1-ivp
                    # 降順←(6)
                    for ihp in range(twidth0-1, -nstep, -nstep):
                        if ihp < 0:
                            ihp = 0
                        xp = ihp - hgh
                        ah = np.sqrt(xp*xp + yp*yp) / rv
                        cnh = np.cos(ah)
                        snh = np.sin(ah)
                        av = np.arctan2(yp, xp)
                        cnv = np.cos(av)
                        snv = np.sin(av)
                        if fv != 0:
                            ah = np.arctan2(snh, (cnh - fv))
                            cnh = np.cos(ah)
                            snh = np.sin(ah)
                        iha = twidth0-1-ihp
                        i = iha + twidth0*iva
                        tcp[i][0] = snh*cnv
                        tcp[i][1] = snh*snv
                        tcp[i][2] = cnh
#
            # i = 0
            # for iyp in reversed(range(theight0)):
            #     yp = iyp - (theight0-1.0)/2.0 - gbias
            #     # 降順←(6)
            #     for ixp in range(twidth0-1, -nstep, -nstep):
            #         if ixp < 0:
            #             ixp = 0
            #         xp = ixp - hgh
            #         ah = np.sqrt(xp*xp + yp*yp) / rv
            #         cnh = np.cos(ah)
            #         snh = np.sin(ah)
            #         av = np.arctan2(yp, xp)
            #         cnv = np.cos(av)
            #         snv = np.sin(av)
            #         if fv != 0:
            #             ah = np.arctan2(snh, (cnh - fv))
            #             cnh = np.cos(ah)
            #             snh = np.sin(ah)
            #         tcp[i][0] = snh*cnv
            #         tcp[i][1] = snh*snv
            #         tcp[i][2] = cnh
            #         pdb.set_trace()
            #         i += 1
                # End of ihp-Loop
            # End of ivp-Loop
        # Break of 倒立 6-0-1
    # Break of 球面補正 6-0-(0/1)
    elif hosei_mode == 1:  # 円筒面補正(円筒面射影) 6-1-(0/1)
        if upright:  # 直立 6-1-0
            # rint('load_tcp_cv/センタービュー/円筒面補正(円筒面射影)/直立 6-1-0')
            # 昇順↓
            i = 0
            for ivp in range(theight0):
                yp = ivp - (theight0 - 1.0) / 2.0 - gbias
                c2 = np.arctan2(yp, rv)
                cos2 = np.cos(c2)
                sin2 = np.sin(c2)
                iva = ivp
                # 昇順→(7)
                for ihp in range(0, twidth0-1+nstep, nstep):
                    if ihp > twidth0 - 1:
                        ihp = twidth0 - 1
                    xp = ihp - hgh
                    c1 = xp / rv
                    cos1 = np.cos(c1)
                    sin1 = np.sin(c1)
                    if fv != 0:
                        sin2 = np.abs(sin1)
                        c1 = np.arctan2(sin1, (cos1-fv))
                        cos1 = np.cos(c1)
                        sin1 = np.sin(c1)
                        c2 = np.arctan2(yp*np.abs(sin1), rv*sin2)
                        cos2 = np.cos(c2)
                        sin2 = np.sin(c2)
                    iha = ihp
                    i = iha + twidth0*iva
                    tcp[i][0] = sin1*cos2
                    tcp[i][1] = sin2
                    tcp[i][2] = cos1*cos2
                    # pdb.set_trace()
                    i += 1
                # End of ihp-Loop
            # End of ivp-Loop
        # Break of 直立 6-1-0
        else:  # 倒立 6-1-1
            # rint('load_tcp_cv/センタービュー/円筒面補正(円筒面射影)/倒立 6-1-1')
            i = 0
            # 降順↑
            for ivp in reversed(range(theight0)):
                yp = ivp - (theight0 - 1.0)/2.0 - gbias
                c2 = np.arctan2(yp, rv)
                cos2 = np.cos(c2)
                sin2 = np.sin(c2)
                iva = theight0-1-ivp  # new
                # 降順←(8)
                for ihp in range(twidth0-1, -nstep, -nstep):
                    if ihp < 0:
                        ihp = 0
                    xp = ihp - hgh
                    c1 = xp / rv
                    cos1 = np.cos(c1)
                    sin1 = np.sin(c1)
                    if fv != 0:
                        sin2 = np.abs(sin1)
                        c1 = np.arctan2(sin1, (cos1 - fv))
                        cos1 = np.cos(c1)
                        sin1 = np.sin(c1)
                        c2 = np.arctan2(yp*np.abs(sin1), rv*sin2)
                        cos2 = np.cos(c2)
                        sin2 = np.sin(c2)
                    iha = twidth0-1-ihp  # new
                    i = iha + twidth0*iva
                    tcp[i][0] = sin1*cos2
                    tcp[i][1] = sin2
                    tcp[i][2] = cos1*cos2
                    # pdb.set_trace()
                    i += 1
                # End of ihp-Loop
            # End of ivp-Loop
        # Break of 倒立 6-1-1
    # Break of 円筒面補正 6-1-(0/1)
# End of load_tcp_cv () センタービュー 6


#
# rdinit_sub()
#
def rdinit_sub(tcp, stm, nstep, twidth0, theight0, rhagv, gmp, gir):
    # global gbias
    # twidth0 = gsz.getval()[0]  # スクリーン幅[pxl]
    # theight0 = gsz.getval()[1]  # スクリーン高[pxl]
    vgh = (theight0 - 1) / 2.0
    view_mode = gmp.getval()[0]  # ビューモード
    hosei_mode = gmp.getval()[1]  # 補正モード
    # upright = gmp.getval()[2]  # 直立/倒立フラグ
    mirror_mode = gmp.getval()[3]  # 実像/虚像(鏡像)フラグ
    work = rhagv - np.radians(6.)
    gckl = 1.0 if np.degrees(work) > 180. else np.power((np.pi/work),4)
    zv = twidth0 / rhagv
    rv = zv * gckl

    def factor_v(agh, ckl):
        if ckl == 1.0:
            ret = 0.0
        else:  # 仮想深度 Zv の ckl 倍
            c1 = agh/2  # α [rad]1.3962634015954636 => 80 deg相当
            c2 = c1/ckl  # α2 [rad]0.8716803677693978 => 49.944 deg相当
            work = np.sin(c2)/np.sin(c1)
            ret = np.cos(c2) - work*np.cos(c1)
        return ret

    fv = factor_v(rhagv, gckl) # fv

    if view_mode == 3:
        # リングビューモード(Ring View Mode) 3-(0/(1)-(0/1)
        if hosei_mode == 0:  # リングビューモード/球面補正 3-0-(0/1)
            # rint('rdinit_sub/リングビューモード/球面補正 3-0-(0/1)')
# koko kara new
            if stm[2,2] > 1.0:
                ang = np.pi/2.0
            elif stm[2,2] < -1.0:
                ang = -np.pi/2.0
            elif mirror_mode == False:
                ang = np.arcsin(stm[2,2])
            else:
                ang = -np.arcsin(stm[2,2])
            cosmin = np.cos(np.radians(gir.getval()[0]))
            i = 0
            for iyp in reversed(range(theight0)):
                yp = iyp - vgh
                ah = yp/rv
                cnh = np.cos(ah)
                snh = np.sin(ah)
                if fv != 0:
                    ah = np.arctan2(snh, cnh-fv)
                    cnh = np.cos(ah)
                    snh = np.sin(ah)
                c2 = snh
                c3 = cnh
                if mirror_mode == False:
                    c33 = -c2*np.cos(ang)+c3*np.sin(ang)
                else:
                    c33 = -c2*np.cos(ang)-c3*np.sin(ang)
                if c33 > cosmin:
                    break
                i = i + 1
            if i > theight0-1:
                i = 0
            gbias = i
# koko made new
        if hosei_mode == 1:  # リングビューモード/円筒面補正 3-1-(0/1)
            # rint('リングビューモード/円筒面補正 3-1-(0/1)')
# koko kara old
            if stm[2,2] > 1.0:
                ang = np.pi/2.0
            elif stm[2,2] < -1.0:
                ang = -np.pi/2.0
            elif mirror_mode == False:
                ang = np.arcsin(stm[2,2])
            else:
                ang = -np.arcsin(stm[2,2])
            cosmin = np.cos(np.radians(gir.getval()[0]))
            rp = rv*(1.0 - fv)
            i = 0
            for iyp in reversed(range(theight0)):
                yp = iyp - vgh
                av = np.arctan2(yp, rp)
                c2 = np.sin(av)
                c3 = np.cos(av)
                if mirror_mode == False:
                    c33 = -c2*np.cos(ang)+c3*np.sin(ang)
                else:
                    c33 = -c2*np.cos(ang)-c3*np.sin(ang)
                if c33 > cosmin:
                    break
                i = i + 1
            if i > theight0-1:
                i = 0
            gbias = i
#
            # zp = rv*(1.0 - fv)  # 180.4
            # yp = -(zp*cosmin)/np.sqrt(1.0 - cosmin*cosmin)  # 104.156
            # yp = np.abs(yp)
            # pdb.set_trace()
# koko made old
        # End of H-Mode-Quary
        # 方向余弦(センタービューモード/リングビューモード共通)
        load_tcp_rv(tcp, rv, fv, twidth0, theight0, nstep, gbias, gmp)
    elif view_mode == 6:
        # センタービューモード(Center View Mode) 6-(0/(1)-(0/1)
        # rint('$ センタービューモード(Center View Mode) 6-(0/1)-(0/1)')
# koko kara newest
        if stm[2,2] > 1.0:
            ang = np.pi/2.0
        elif stm[2,2] < -1.0:
            ang = -np.pi/2.0
        else:
            ang = np.arcsin(stm[2,2])
        if mirror_mode:
            ang = -ang
        xp = 0.5
        cosmin = np.cos(np.radians(gir.getval()[0]))
        i = 0
        for iyp in reversed(range(theight0)):
            yp = iyp - vgh
            ah = np.sqrt(xp*xp + yp*yp)/rv
            cnh = np.cos(ah)
            snh = np.sin(ah)
            av = np.arctan2(yp, xp)
            snv = np.sin(av)
            if fv != 0:
                ah = np.arctan2(snh, cnh-fv)
                cnh = np.cos(ah)
                snh = np.sin(ah)
            c2 = snh*snv
            c3 = cnh
            if mirror_mode == False:
                c33 = -c2*np.cos(ang)+c3*np.sin(ang)
            else:
                c33 = -c2*np.cos(ang)-c3*np.sin(ang)
            if c33 > cosmin:
                break
            i = i + 1
        if i > theight0-1:
            i = 0
        gbias = i
# koko mad newest
        # 方向余弦(センタービューモード/リングビューモード共通)
        load_tcp_cv(tcp, rv, fv, twidth0, theight0, nstep, gbias, gmp)
    return gbias
    # End of ViewMode-Quary
# End of rdinit_sub ()


########################################################################
#   put_pixel() / 画像のpxl座標にtupleカラーコードを書込 tkinter
########################################################################
def put_pixel(image, x, y, *tup_code):
    ((rgb),) = tup_code
    hex_code = "#{:02x}{:02x}{:02x}".format(*rgb)
    image.put(hex_code, to=(x, y))


########################################################################
# ソース画像のpxl座標のtupleカラーコードを読込
########################################################################
def getcolor_fast(stupcd2: np.ndarray, x: float, y: float):
    x0 = int(x) // 2
    y0 = int(y) // 2

    dx = int((x % 2) * 4)  # 0〜1.99 → 0〜3
    dy = int((y % 2) * 4)

    row = y0 * 2 + dy
    col = x0 * 2 + dx

    row = min(row, stupcd2.shape[0] - 1)
    col = min(col, stupcd2.shape[1] - 1)
    # row = max(0, min(row, stupcd2.shape[0] - 1))
    # col = max(0, min(col, stupcd2.shape[1] - 1))

    return stupcd2[row, col]
# End of getcolor_fast ()


########################################################################
# ソース画像の座標のtupleカラーコードと座標を読込
########################################################################
def getcolorpx_fast(stupcd2: np.ndarray, x: float, y: float):
    x0 = int(x) // 2
    y0 = int(y) // 2

    dx = int((x % 2) * 2)  # 0〜1.99 → 0〜3
    dy = int((y % 2) * 2)

    row = y0 * 2 + dy
    col = x0 * 2 + dx

    row = min(row, stupcd2.shape[0] - 1)
    col = min(col, stupcd2.shape[1] - 1)

    # オフセット定義（dx, dy）→ 座標補正（4分割に対応）
    offset_table = {
        (0, 0): (0.0, 0.0),
        (1, 0): (0.25, 0.0),
        (2, 0): (0.5, 0.0),
        (3, 0): (0.75, 0.0),
        (0, 1): (0.0, 0.25),
        (1, 1): (0.25, 0.25),
        (2, 1): (0.5, 0.25),
        (3, 1): (0.75, 0.25),
        (0, 2): (0.0, 0.5),
        (1, 2): (0.25, 0.5),
        (2, 2): (0.5, 0.5),
        (3, 2): (0.75, 0.5),
        (0, 3): (0.0, 0.75),
        (1, 3): (0.25, 0.75),
        (2, 3): (0.5, 0.75),
        (3, 3): (0.75, 0.75),
    }

    offset_x, offset_y = offset_table[(dx, dy)]
    px = x0 + offset_x
    py = y0 + offset_y

    return stupcd2[row, col], (px, py)
# Enf of getcolorpx_fast ()


########################################################################
# ハイスピードモード thexcd(Hex Code) <- shexcd(Hex Code)
########################################################################
def hosei_sub_hs(ttupcd, stupcd1, tcp, stm, fast, nstep, twidth0, theight0, *params):
    # cdef int nhg, nstep1
    # cdef int ita, itb, k, itcp, ixtb, iytb
    # cdef double ddh, r, rad0, rad1, work
    # cdef double xs, ys, limsup, liminf, cosmin, cosmax
    # cdef double xsa, ysa, xsb, ysb
    # cdef int ixsa, iysa, ixsb, iysb, iyp, ixp, ixr
    # cdef int ihp, ixta, iyta, ixa, iya
    #
    ((dd_u, gbias, mp, limit, color),) = params
    limsup, liminf = limit[0], limit[1]
    fg, bg = color[0], color[1]
    ddh = (dd_u-1) / 2.0  # 全方位画像半径
    # rad1 = np.floor(ddh)
    rad1 = ddh
    # if mp[0] == 6:
    #    rad1 -= 2
    rad0 = 0.0
    cosmin = np.cos(np.radians(limsup))  # 例:-0.50=cos(120deg)
    cosmax = np.cos(np.radians(liminf))  # 例:+0.87=cos(30deg)
    cp = np.zeros((3, 3))
    nhg = int((twidth0-2) / nstep + 2)
    nstep1 = (twidth0-1) % nstep
    ita = 0  # 0列用インデックス
    # ixsa, iysa, ixsb, iysb = 0, 0, 0, 0ß
    for iyp in range(theight0):
        yp = iyp - (theight0-1.0)/2.0 - gbias
        ihp = 0
        #
        # インターバルの始端列 a(0列)
        itcp = ihp + twidth0*iyp
        cp = stm @ tcp[itcp].reshape(-1, 1)
        k = int(abs(cp[2, 0]) * 1000)
        if cp[2, 0] > 0:
            work = fast[0, k]
        else:
            work = fast[1, k]
        xs = cp[0, 0]*work
        ys = cp[1, 0]*work
        xsa = (xs+ddh)
        ysa = (ys+ddh)
        ixta = ita % twidth0
        iyta = ita // twidth0
        if cp[2, 0] > cosmin:
            # 撮像範囲内: 表示可
            ia = True
        else:
            # 撮像範囲外: 表示不可
            ia = False
        ixsa = round(xsa)
        iysa = round(ysa)
        if ixsa > dd_u-1:
            ixsa = dd_u-1
        if iysa > dd_u-1:
            iysa = dd_u-1
        if ia:
            # 撮像範囲内: Target Image <- Source Image
            ttupcd[iyta][ixta] = stupcd1[iysa][ixsa]  #r
            # tup_cod = s image.getpixel((ixsa, iysa))
            # put_pixel(timage, ixta, iyta, tup_cod)
        else:
            # 撮像範囲外: Target Image <- Background Color
            ttupcd[iyta][ixta] = stupcd1[1][1]  #r
            # put_pixel(timage, ixta, iyta, bg)
        # End of a-column
        #
        ihp = 0
        for ixr in range(1, nhg):
            mstep = ixr*nstep
            if mstep > twidth0-1:
                mstep = twidth0-1
            itb = twidth0*iyp + mstep
            ihp += nstep
            if ihp > twidth0 - 1:
                ihp = twidth0 - 1
            # インターバルの終端列 b(次の始端列 a)
            itcp = ihp + twidth0*iyp
            cp = stm @ tcp[itcp].reshape(-1, 1)  # ティルト
            k = int(abs(cp[2, 0]) * 1000)
            if cp[2, 0] > 0:
                work = fast[0, k]
            else:
                work = fast[1, k]
            xs = cp[0, 0]*work
            ys = cp[1, 0]*work
            xsb = (xs+ddh)
            ysb = (ys+ddh)
            ixtb = itb % twidth0
            iytb = itb // twidth0
            if cp[2, 0] > cosmin:
                # 撮像範囲内(方向余弦判定): 表示可
                ib = True
            else:
                # 撮像範囲外(方向余弦判定): 表示不可
                ib = False
            if xsb >= dd_u-1+0.5:
                xsb = dd_u-1+0.4
            if ysb >= dd_u-1+0.5:
                ysb = dd_u-1+0.4
            ixsb = round(xsb)
            iysb = round(ysb)
            if ib:
                # 撮像範囲内(方向余弦判定)
                # Source Pointer が Source Image の中央孔内外か方向余弦判定
                if cp[2, 0] < cosmax:
                    # 中央孔外: Target Image <- Source Image
                    ttupcd[iytb][ixtb] = stupcd1[iysb][ixsb]
                    # tup_cod = s image.getpixel((ixsb, iysb))
                    # put_pixel(t image, ixtb, iytb, tup_cod)
                else:
                    # 中央孔内: Target Image <- Foreground Color
                    ttupcd[iytb][ixtb] = fg  # stupcd1[0][0]
                    # put_pixel(t image, ixtb, iytb, fg)
                    if rad0 == 0.0:
                        # 半径判定のための半径
                        rad0 = abs(complex(xs, ys))
            else:
                # 撮像範囲外(方向余弦判定): Target Image <- Background Color
                ttupcd[iytb][ixtb] = bg  # stupcd1[1][1]
                # put_pixel(timage, ixtb, iytb, bg)
            # End of b-column
            #
            # a->b間線形補間
            if ia or ib:
                # 列両端ab共に範囲内
                if mstep-(ixr-1)*nstep == nstep:
                    # 通常列(列数 == nstep)
                    iw = nstep
                else:
                    # 余剰列(列数 == nstep1)
                    iw = nstep1
                # End of nstep-Quary
                if iw > 1:
                    mx = mxd = (ixsb - ixsa) / iw
                    my = myd = (iysb - iysa) / iw
                    for ixp in range((ixr-1)*nstep+1, mstep+1):
                        xsa = ixsa + mx
                        mx += mxd
                        ysa = iysa + my
                        my += myd
                        r = np.floor(abs(complex(xsa-ddh, ysa-ddh))-0.5)
                        # Source Pointer が Source Image が表示可か半径判定
                        if rad1 < r:
                            # 撮像範囲外: Target Image <- Background Color
                            ttupcd[iyp][ixp] = bg  # stupcd1[1][1]
                            # put_pixel(timage, ixp, iyp, bg)
                        elif r < rad0:
                            # 中央孔内: Target Image <-  Foreground Color
                            ttupcd[iyp][ixp] = fg  # stupcd1[0][0]  #r
                            # put_pixel(timage, ixp, iyp, fg)
                        else:
                            # 撮像範囲内 & 中央孔外: Target Image <- Source Image
                            ttupcd[iyp][ixp] = stupcd1[round(ysa)][round(xsa)]  #r
                            # tup_cod = s image.getpixel((round(xsa),round(ysa)))
                            # put_pixel(timage, ixp, iyp, tup_cod)  # a < range
                    # End of iw-Loop
                # End of nstep1-Quary
            else:
                # 撮像範囲外(方向余弦判定): Target Image <- Background Color
                for ixp in range((ixr-1)*nstep, mstep):
                    ttupcd[iyp][ixp] = bg  # stupcd1[1][1]
                    # put_pixel(timage, ixp, iyp, bg)
                # End of ixp-Loop
            # End of [ia,ib]-Quary
            ia, ixsa, iysa = ib, ixsb, iysb
        # End of ixr-Loop(H-Loop)
        ita += twidth0  # 0列用インデックス改行
    # End of iyp-Loop(V-Loop)
# End of _hosei_sub_hs


########################################################################
# スクリーンに射影 thexcd <- stupcd
# ハイレゾモード
########################################################################
def hosei_sub_hr(ttupcd, stupcd2, tcp, stm, fast, nstep, twidth0, theight0, *params):
    ((dd_u, gbias, mp, limit, color),) = params
    limsup, liminf = limit[0], limit[1]
    fg, bg = color[0], color[1]
    factor = 2.0
    ddh = (dd_u-1) / 2.0  # 全方位画像半径
    cosmin = np.cos(np.radians(limsup))  # 例:-0.50=cos(120deg)
    cosmax = np.cos(np.radians(liminf))  # 例:+0.87=cos(30deg)
    nhg = int((twidth0-2) / nstep + 2)
    nstep1 = (twidth0-1) % nstep
    # i0 = 0  # 0列用インデックス(0,640,1280,...)
    i2 = 0  # 1～(twidth0-1)列用インデックス
    ixsb, iysb = 0, 0
    for iyp in range(theight0):
        yp = iyp - (twidth0-1.0)/2.0 - gbias
        # 0列(i0)(0,640,1280,...)
        ihp = 0
        itcp = ihp + twidth0*iyp
        cp = stm @ tcp[itcp].reshape(-1, 1)  # ティルト
        k = int(abs(cp[2, 0]) * 1000)
        if k > 1000-1:
            k = 1000-1
        if cp[2, 0] > 0:
            work = fast[0, k]
        else:
            work = fast[1, k]
        xs = cp[0, 0]*work
        ys = cp[1, 0]*work
        xsa = (xs+ddh)*factor
        ysa = (ys+ddh)*factor
        ixta = i2 % twidth0
        iyta = i2 // twidth0
        if cp[2, 0] > cosmin:
            # 範囲内で表示可
            ia = True
        else:
            # 範囲外で表示不可
            ia = False
        if ia:
            # インターバルの列始端 a
            ttupcd[iyta][ixta], (ixsa, iysa) = getcolorpx_fast(stupcd2, xsa, ysa)
        else:
            # 撮像範囲外(方向余弦判定): Target Image <- Background Color
            ttupcd[iyta][ixta] = bg  # stupcd2[1][1]
        # End of ia-Quary
        # 1列以降
        for ixr in range(1, nhg):
            mstep = ixr*nstep
            if mstep > twidth0-1:
                mstep = twidth0-1
            i1 = twidth0*iyp + mstep
            ihp += nstep
            if ihp > twidth0 - 1:
                ihp = twidth0 - 1
            itcp = ihp + twidth0*iyp
            cp = stm @ tcp[itcp].reshape(-1, 1)  # ティルト
            k = int(abs(cp[2, 0]) * 1000)
            if k > 1000-1:
                k = 1000-1
            if cp[2, 0] > 0:
                work = fast[0, k]
            else:
                work = fast[1, k]
            xs = cp[0, 0]*work
            ys = cp[1, 0]*work
            xsb = (xs+ddh)*factor
            ysb = (ys+ddh)*factor
            ixtb = i1 % twidth0
            iytb = i1 // twidth0
            if cp[2, 0] > cosmin:
                # 範囲内で表示可
                ib = True
            else:
                # 範囲外で表示不可
                ib = False
            if ib:
                # インターバルの列終端 b(次の列始端 a)
                if cp[2, 0] < cosmax:
                    ttupcd[iytb][ixtb], (ixsb, iysb) = getcolorpx_fast(stupcd2, xsb, ysb)
                else:
                    # 中央孔内: Target Image <- Foreground Color
                    ttupcd[iytb][ixtb] = fg  # stupcd2[0][0]
            else:
                # 撮像範囲外(方向余弦判定): Target Image <- Background Color
                ttupcd[iytb][ixtb] = bg  # stupcd2[1][1]
            # End of ib-Quary
            # a->b間線形補間
            if all([ia,ib]) and nstep > 1:
                # 列両端ab共に範囲内
                if mstep-(ixr-1)*nstep == nstep:
                    # 通常列(列数 == nstep)
                    iw = nstep
                else:
                    # 余剰列(列数 == nstep1)
                    iw = nstep1
                # End of nstep-Quary
                if iw > 1:
                    mx = mxd = (ixsb - ixsa) / iw
                    my = myd = (iysb - iysa) / iw
                    for ixp in range((ixr-1)*nstep+1, mstep+1):
                        xsa = factor * (ixsa + mx)
                        mx += mxd
                        ysa = factor * (iysa + my)
                        my += myd
                        if cp[2, 0] < cosmax:
                            ttupcd[iyp][ixp] = getcolor_fast(stupcd2, xsa, ysa)
                        else:
                            # 中央孔内: Target Image <- Foreground Color
                            ttupcd[iyp][ixp] = fg  # stupcd2[0][0]
                    # End of iw-Loop
                # End of nstep1-Quary
            elif nstep > 1:
                # 撮像範囲外(方向余弦判定): Target Image <- Background Color
                for ixp in range((ixr-1)*nstep, mstep):
                    ttupcd[iyp][ixp] = bg  # stupcd2[1][1]
                    # put_pixel(timage, ixp, iyp, bg)
                # End of ixp-Loop
            # End of [ia,ib]-Quary
            ia, ixsa, iysa = ib, ixsb, iysb
        # End of ixr-Loop(H-Loop)
        i2 += twidth0  # 1～(twidth0-1)列用インデックス改行
        # i0 += nhg  # 0列用インデックス改行
    # End of iyp-Loop(V-Loop)
# End of hosei_sub_hr


########################################################################
# プリプロセススタート
########################################################################
def pre_process(template_key):
    if template_key in preprocess_cache:
        return preprocess_cache[template_key]

    preprocess_cache.clear()

    file_path = os.path.join("templates", f"{template_key}.json")
    if not os.path.exists(file_path):
        raise ImageFileNotFoundError(
            f"The Configuration File ({template_key}.json) is not found."
        )

    if Footstep:
        print('----- start of pre_process() -----', flush = True)

    # 設定ファイルを読込(setting.json)
    (
        gmp,
        gops,
        gir,
        gva,
        gsz,
        gim,
        gmc,
        gft
    ) = get_setting(file_path)

    if Footstep:
        print('----- configration file loaded -----', flush = True)

    def hidden_setting(gft):
        cleaned = []
        print(f"{gft.getval()[0] = }", flush = True)  # 消さない
        if len(gft.getval()[0]) > 0:
            row = next(csv.reader([gft.getval()[0]]))
            if len(row) > 0:
                for i, col in enumerate(row):
                    try:
                        val = float(col.strip()) if col.strip() else 0.0
                    except ValueError:
                        break
                        raise ValueError(f"{i+1}番目の値「{col}」が数値として不正です")
                    cleaned.append(val)
        return cleaned

    # 設定情報を変数に代入
    upright = gmp.getval()[2]  # 直立/倒立フラグ
    mirror_mode = gmp.getval()[3]  # 実像/虚像(鏡面)フラグ
    projection = gmp.getval()[4]  # 射影モード

    # 初期視線方向
    rvang_z = np.radians(gva.getval()[2])
    rvang_x = np.radians(gva.getval()[0])

    # ソース原画を読込
    image_path = "static/" + gim.getval()[0]
    if not os.path.exists(image_path):
        raise ImageFileNotFoundError(
            f"The image file ({gim.getval()[0]}) was not found."
        )
    try:
        simage = Image.open(image_path).convert("RGB")
    except PILUnidentifiedImageError:
        raise ImageUnidentifiedError(
            f"The file ({gim.getval()[0]}) could not be identified as an image."
        )
    # ソース画像の正方形チェック
    if (simage.width != simage.height):
        raise OutOfRangeValueError("Source image must be square")

    # ソース画像のサイズ限定
    # fsize = False
    # fshrink = True
    # cleaned = hidden_setting(gft)
    # if len(cleaned) > 5:
    #     if cleaned[5] > 0.0:
    #         fshrink = False
    # if fshrink and simage.width <= 1024:
    #     simage = simage.resize((1024, 1024), Image.LANCZOS)
    #     dd_u = 1024
    # else:
    #     dd_u = simage.width
    #     fsize = True
    # if Footstep or fsize:
    #     print(f"{dd_u = }")

    if Footstep:
        print('----- Source image loaded -----', flush = True)

########################################################################
# Source Image と Target Image をNumPy配列に変換
########################################################################
    # upscale image ----------------------------------------------------
    def upscale_with_interpolation(img):
        img = img.astype(np.uint16)
        H, W, C = img.shape
        H2, W2 = H * 2 - 1, W * 2 - 1
        up = np.zeros((H2, W2, C), dtype = np.uint16)
        # 元画素を配置
        up[::2, ::2] = img
        # 横方向の線形補間
        up[::2, 1::2] = (img[:, :-1] + img[:, 1:]) // 2
        # 縦方向の線形補間
        up[1::2, ::2] = (img[:-1, :] + img[1:, :]) // 2
        # 中央（面積）補間
        up[1::2, 1::2] = (img[:-1, :-1] + img[1:, :-1] + img[:-1, 1:] + img[1:, 1:]) // 4
        # uint8 にキャスト
        return np.clip(up, 0, 255).astype(np.uint8)


    def safe_upscale_with_pillow(img: Image.Image, factor: int = 2) -> Image.Image:
        new_width = img.width * factor
        new_height = img.height * factor
        if new_width * new_height > MAX_PIXELS:
            logging.warning(f"Upscaled image ({new_width}x{new_height}) exceeds limit; resizing to {MAX_PIXELS} pixels")
            scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, resample=Image.LANCZOS)
            new_width = new_size[0] * factor
            new_height = new_size[1] * factor
        return img.resize((new_width, new_height), resample=Image.LANCZOS)


    def will_overflow(width, height, channels=3, dtype=np.uint16) -> bool:
        bytes_per_pixel = np.dtype(dtype).itemsize
        total_bytes = width * height * channels * bytes_per_pixel
        return total_bytes > MAX_PIXELS


    def safe_upscale(img_pil: Image.Image) -> np.ndarray:
        width, height = img_pil.size
        if will_overflow(width * 2 - 1, height * 2 - 1):  # アップスケール後のサイズで判定
            logging.warning("Using Pillow for safe upscaling due to memory concern")
            img_pil = safe_upscale_with_pillow(img_pil)
            return np.array(img_pil, dtype=np.uint8)
        else:
            img_np = np.array(img_pil, dtype=np.uint8)
            return upscale_with_interpolation(img_np)


    dd_u = simage.width
    resized = True if dd_u <= 1024 else False

    try:
        stupcd1 = np.array(simage, dtype = np.uint8)
    except MemoryError:
        if resized:
            raise
        logging.warning("MemoryError: trying to resize image for recovery")
        gc.collect()
        simage = simage.resize((1024, 1024), Image.LANCZOS)
        resized = True
        dd_u = 1024
        try:
            stupcd1 = np.array(simage, dtype=np.uint8)
        except MemoryError:
            logging.error("Recovery failed after resizing. Exiting.")
            raise

    try:
        stupcd2 = safe_upscale(simage)
    except Exception as e:
        logging.error(f"Upscaling failed: {e}")
        raise

    # ソース画像をNumPy配列に変換(バッファーとして)
    # stupcd1 = np.array(simage, dtype=np.uint8)
    # stupcd2 = upscale_with_interpolation(stupcd1)

    if Footstep:
        print('----- Source RGB-files created -----', flush = True)

    # ターゲット画像サイズ
    twidth1 = gsz.getval()[0]  # 画像幅(軽量ハイスピード) [px]
    theight1 = gsz.getval()[1]  # 画像高さ(軽量ハイスピード) [px]
    twidth2 = gsz.getval()[3]  # 画像幅(ハイレゾリューション) [px]
    theight2 = gsz.getval()[4]  # 画像高さ(ハイレゾリューション) [px]

    atn_table = {
        1: 1.0,
        2: np.sqrt(2.),
        3: np.sqrt(3.),
        4: 2.0,
        5: np.sqrt(5.),
        6: np.sqrt(6.),
        7: np.sqrt(7.),
        8: 2.*np.sqrt(2.)
    }
    cleaned = hidden_setting(gft)
    fsize = False
    if len(cleaned) > 4:
        cl4 = int(cleaned[4])
        if 1 <= cl4 and cl4 <= 8:
            twidth1 = round(twidth2/atn_table[(cl4)]/2.)*2
            theight1 = round(theight2*twidth1/twidth2)
            fsize = True

    if Footstep or fsize:
        print(f"{twidth1 = } {theight1 = }", flush = True)
        print(f"{twidth2 = } {theight2 = }", flush = True)

    # ターゲット画像を作成
    ttupcd1 = np.zeros((theight1, twidth1, 3), dtype = np.uint8)
    ttupcd2 = np.zeros((theight2, twidth2, 3), dtype = np.uint8)

    if Footstep:
        print('----- Target RGB-files created -----', flush = True)

########################################################################
# 設定値を読込/内部変数を設定
########################################################################
    # 水平画角を読込([rad]<-[deg])
    rhagv = np.radians(gsz.getval()[2])  # 水平画角 [rad](動的値)

    # ステップ距離 [px]
    nstep1 = 4  # ハイスピードモード
    nstep2 = 1  # ハイレゾモード

    cleaned = hidden_setting(gft)
    fstep = False
    if len(cleaned) > 2:
        cl2 = int(cleaned[2])
        if 3 <= cl2 and cl2 <= 16:
            nstep1 = cl2
            fstep = True
    if len(cleaned) > 3:
        cl3 = int(cleaned[3])
        if 1 < cl3 and cl3 <= 2:
            nstep2 = cl3
            fstep = True
    if Footstep or fstep:
        print(f"{nstep1 = } {nstep2 = }", flush = True)

    # 方向余弦配列を生成/ゼロクリア
    tcp1 = np.zeros((twidth1*theight1, 3))  # 画像サイズ(軽量ハイスピード) [px]
    tcp2 = np.zeros((twidth2*theight2, 3))  # 画像サイズ(ハイレゾリューション) [px]

    # 方向余弦LUTを作成/関数set_fast()をコール
    #  テーブルゼロクリア(1000:テーブルサイズ)
    fast = np.zeros((2, 1000))
    # 係数チャージ(光学系に依存)
    set_fast(fast, dd_u, projection, gops, gir)

    # 基本マトリック(正像,実像: φ=0, ψ=90)
    stm = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    #  ミラーの場合 座標変換行列の3列要素を正負反転
    if mirror_mode:
        # 虚像(鏡像)に変換
        stm[:,2] = -stm[:,2]
        # ([[1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])

    # z-軸周り [rad](水平方向移動に相当)
    agh = rvang_z
    cnh = np.cos(agh)
    snh = np.sin(agh)
    dmath = np.array([[cnh, -snh, 0.], [snh, cnh, 0.], [0., 0., 1.]])
    stm = dmath @ stm  # パン

    # x-軸周り [rad](鉛直方向移動に相当)
    if upright ^ mirror_mode:
        agv = -rvang_x
        gdtx = -gva.getval()[0]  # [deg]
    else:
        agv = rvang_x
        gdtx = gva.getval()[0]  # [deg]
    cnv = np.cos(agv)
    snv = np.sin(agv)
    dmatv = np.array([[1., 0., 0.], [0., cnv, snv], [0., -snv, cnv]])
    stm = stm @ dmatv  # ティルト

    gbias = 0  # = rdinit_sub(tcp2, stm, nstep2, twidth2, theight2, rhagv, gmp, gir)
    params = (
        dd_u,
        gbias,
        gmp.getval(),
        gir.getval(),
        gmc.getval()
    )
    # 初期画像(RGBカラー配列)ハイレゾリューションモード
    # hosei_sub_hr(ttupcd2, stupcd2, tcp2, stm, fast, nstep2, twidth2, theight2, params)

    if Footstep:
        print('----- Initial target image prepared -----', flush = True)

    # ---------------------------------------------------------------
    # 水平/鉛直方向の増分角度
    delta_h = 1.0
    delta_v = 1.0

    cleaned = hidden_setting(gft)
    fdelta = False
    if len(cleaned) > 0:
        if cleaned[0] > 0.0:
            delta_h = cleaned[0]
            fdelta = True
    if len(cleaned) > 1:
        if cleaned[1] > 0.0:
            delta_v = cleaned[1]
            fdelta = True

    delta_h = 90.0/int(90.0/delta_h)  # 90 [deg] の約数化
    grangle_h = np.radians(delta_h)  # [rad]
    cnh = np.cos(grangle_h)
    snh = np.sin(grangle_h)

    # Φの増加方向
    dmathp = np.array([
        [cnh, -snh, 0.],
        [snh, cnh, 0.],
        [0., 0., 1.]
    ])
    # Φの減少方向
    dmathm = np.array([
        [cnh, snh, 0.],
        [-snh, cnh, 0.],
        [0., 0., 1.]
    ])

    delta_v = 90.0/int(90.0/delta_v)
    grangle_v = np.radians(delta_v)  # [rad]
    cnv = np.cos(grangle_v)
    snv = np.sin(grangle_v)

    # Ψの増加方向
    dmatvp = np.array([
        [1., 0., 0.],
        [0., cnv, snv],
        [0., -snv, cnv]
    ])
    # Ψの減少方向
    dmatvm = np.array([
        [1., 0., 0.],
        [0., cnv, -snv],
        [0., snv, cnv]
    ])

    if Footstep or fdelta:
        print(f"{delta_h = }")
        print('dmathm =\n', dmathm, flush = True)
        print(f"{delta_v = }")
        print('dmatvm =\n', dmatvm, flush = True)

    pm90 = -90 if mirror_mode else 90

    preprocess_cache[template_key] = {
        'stm_init': stm,  # 基本マトリックス初期値
        'rhagv_init': rhagv,  # 水平画角初期値
        'tcp1': tcp1,  # 方向余弦(Variable)
        'tcp2': tcp2,  # 方向余弦(Variable)
        'fast': fast,  # 方向余弦配列(Constant)
        'stupcd1': stupcd1,  # ハイスピード用ソース画像のRGB配列(Constant)
        'stupcd2': stupcd2,  # ハイレゾ用ソース画像のRGB配列(Constant)
        'ttupcd1': ttupcd1,  # ターゲット画像のRGB配列(Variable)
        'ttupcd2': ttupcd2,  # ターゲット画像のRGB配列(Variable)
        'mirror_mode': mirror_mode,  # 直立/倒立フラグ(Constant)
        'upright': upright,  # 実像/虚像(鏡面)フラグ(Constant)
        'twidth1': twidth1,  # 画像幅(軽量ハイスピード) [px]
        'theight1': theight1,  # 画像高さ(軽量ハイスピード) [px]
        'twidth2': twidth2,  # 画像幅(ハイレゾリューション) [px]
        'theight2': theight2,  # 画像高さ(ハイレゾリューション) [px]
        'gmp': gmp,  # モードパラメーター
        'gir': gir,  # イメージングレンジ
        'gsz': gsz,  # スクリーンサイズ
        'nstep1': nstep1,  # ハイスピードモードのステップ数(Constant)
        'nstep2': nstep2,  # ハイレゾモードのステップ数(Constant)
        'dmathp': dmathp,  # Φの増加方向(Constant)
        'dmathm': dmathm,  # Φの減少方向(Constant)
        'dmatvp': dmatvp,  # Ψの増加方向(Constant)
        'dmatvm': dmatvm,  # Ψの減少方向(Constant)
        'pm90': pm90, # ±90 deg(Constant)
        'params': params
    }

    if Footstep:
        print('----- Parameters cached -----', flush = True)

    session.pop('start_time', None)
    session.pop("stm", None)
    session.pop("rhagv", None)
    session.pop('needs_init', None)
    session["stm"] = stm.tolist()
    session["rhagv"] = rhagv
    session['needs_init'] = False

    if Footstep:
        print('----- End of pre_process() -----', flush = True)

    return preprocess_cache[template_key]
# End of pre_prosses ()


########################################################################
# クライアントからのリクエストに応じてリアルタイムで補正画像を生成する
########################################################################
########################################################################
# マトリックスstmから鉛直角[Ψ]を求める/arctan2()使用(round())
def get_psi_from(stm,mirror_mode):
    a22 = -stm[2,2] if mirror_mode else stm[2,2]
    return(round(np.degrees(np.arctan2(a22, -stm[2,1]))))


########################################################################
# マトリックスstmから水平角[Φ]を求める/arctan2()使用
def get_phi_from(stm):
    return(np.degrees(np.arctan2(stm[1, 0], stm[0, 0])))


########################################################################
# sessionから方向行列stmを読込/session.clear()直後は初期値stm_initで初期化
def get_stm(stm_init):
    if "stm" not in session:
        session["stm"] = stm_init.tolist()
    return np.array(session["stm"])


########################################################################
# sessionから水平画角rhagvを読込/session.clear()直後は初期値rhagv_initで初期化
def get_rhagv(rhagv_init):
    if "rhagv" not in session:
        session["rhagv"] = rhagv_init
    return session["rhagv"]


########################################################################
# Flaskのリクエストコンテキスト
@app.route("/process_image")
def process_image():
    if TimeMMs:
        start_time = session["start_time"]
        print(f"Transmission Time required {time.time() - start_time:.2f} sec.")
        start_time = time.time()
    if Footstep:
        print(f"process_image started.", flush = True)
    # print(f"process_image {template_key} started.")
    effect_level = int(request.args.get("effect", 0))
    if Footstep:
        print('===== effect_level =', effect_level, ' =====', flush = True)

    # クエリで受け取る
    template_key = request.args.get("template")
    if not template_key or template_key not in preprocess_cache:
        return "設定が見つかりません", 400
    
    data = preprocess_cache[template_key]
    stm_init = data['stm_init']
    rhagv_init = data['rhagv_init']  # 水平画角(Variable)
    tcp1 = data['tcp1']  # 方向余弦(Variable)
    tcp2 = data['tcp2']  # 方向余弦(Variable)
    fast = data['fast']  # 方向余弦配列(Constant)
    stupcd1 = data['stupcd1']  # ハイスピード用ソース画像のRGB配列(Constant)
    stupcd2 = data['stupcd2']  # ハイレゾ用ソース画像のRGB配列(Constant)
    ttupcd1 = data['ttupcd1']  # ターゲット画像のRGB配列
    ttupcd2 = data['ttupcd2']  # ターゲット画像のRGB配列
    mirror_mode = data['mirror_mode']  # 直立/倒立フラグ(Constant)
    upright = data['upright']  # 実像/虚像(鏡面)フラグ(Constant)
    twidth1 = data['twidth1']  # 画像幅(軽量ハイスピード) [px]
    theight1 = data['theight1']  # 画像高さ(軽量ハイスピード) [px]
    twidth2 = data['twidth2']  # 画像幅(ハイレゾリューション) [px]
    theight2 = data['theight2']  # 画像高さ(ハイレゾリューション) [px]
    gmp = data['gmp']  # モードパラメーター
    gir = data['gir']  # イメージングレンジ
    gsz = data['gsz']  # スクリーンサイズ
    # nstep1 = data['nstep1']  # ハイスピードモードのステップ数(Constant)
    # nstep2 = data['nstep2']  # ハイレゾモードのステップ数(Constant)
    # dmathp = data['dmathp']  # Φの増加方向(Constant)
    # dmathm = data['dmathm']  # Φの減少方向(Constant)
    # dmatvp = data['dmatvp']  # Ψの増加方向(Constant)
    # dmatvm = data['dmatvm']  # Ψの減少方向(Constant)
    # pm90 = data['pm90']  # ±90 deg(Constant)
    params = data['params']

    # 更新される変数を取得
    stm = get_stm(stm_init)  # stm <- session
    rhagv = get_rhagv(rhagv_init)  # rhagv <- session

    # ------------------------------------------------------------------
    # 初期画面にリセット
    if effect_level == 0:
        session.pop("stm", None)
        session.pop("rhagv", None)
        stm = get_stm(stm_init)  # stm <- stm_init
        rhagv = get_rhagv(rhagv_init)  # rhagv <- rhagv_init
    # ------------------------------------------------------------------
    # ハイレゾモードで表示
    elif effect_level == 1:
        pass
    # ------------------------------------------------------------------
    # ズームイン
    elif effect_level == 7:
        nstep1 = data['nstep1']  # ハイスピードモードのステップ数(Constant)
        if round(np.degrees(rhagv)) > 72:
            rhagv -= np.radians(2.)
            session["rhagv"] = rhagv  # session <- rhagv
        gbias = rdinit_sub(tcp1, stm, nstep1, twidth1, theight1, rhagv, gmp, gir)
    # ------------------------------------------------------------------
    # ズームアウト
    elif effect_level == 9:
        max_agv = gir.getval()[0]*2 if gmp.getval()[0] == 6 else 360
        if round(np.degrees(rhagv)) < max_agv:
            rhagv += np.radians(2.)
            session["rhagv"] = rhagv  # session <- rhagv
        nstep1 = data['nstep1']  # ハイスピードモードのステップ数(Constant)
        gbias = rdinit_sub(tcp1, stm, nstep1, twidth1, theight1, rhagv, gmp, gir)
    # ------------------------------------------------------------------
    # 水平にリセット(高画質)
    elif effect_level == 5:
        cnh = stm[0, 0]
        snh = stm[1, 0]
        stm = np.array([
            [cnh, 0., -snh],
            [snh, 0., cnh],
            [0., -1., 0.]
        ])
        if mirror_mode:
            # 虚像(鏡像)に変換
            stm[:,2] = -stm[:,2]
        session["stm"] = stm.tolist()
        rhagv = get_rhagv(rhagv_init)
    # ------------------------------------------------------------------
    # 水平に移動(高速)
    elif effect_level == 6:
        dmathp = data['dmathp']  # Φの増加方向(Constant)
        dmathm = data['dmathm']  # Φの減少方向(Constant)
        dmath = dmathm if (upright ^ mirror_mode) else dmathp
    elif effect_level == 4:
        dmathp = data['dmathp']  # Φの増加方向(Constant)
        dmathm = data['dmathm']  # Φの減少方向(Constant)
        dmath = dmathp if (upright ^ mirror_mode) else dmathm
    if effect_level in (4, 6):
        stm = dmath @ stm
        session["stm"] = stm.tolist()
    # ------------------------------------------------------------------
    # 鉛直移動(高速)
    elif effect_level == 8:
        dmatvp = data['dmatvp']  # Ψの増加方向(Constant)
        dmatvm = data['dmatvm']  # Ψの減少方向(Constant)
        pm90 = data['pm90']  # ±90 deg(Constant)
        if (
            (not upright and get_psi_from(stm,mirror_mode) == 0)
            or (upright and get_psi_from(stm,mirror_mode) == pm90)
        ):
            dmatv = np.eye(3)
        else:
            dmatv = dmatvm if upright else dmatvp
    elif effect_level == 2:
        dmatvp = data['dmatvp']  # Ψの増加方向(Constant)
        dmatvm = data['dmatvm']  # Ψの減少方向(Constant)
        pm90 = data['pm90']  # ±90 deg(Constant)
        if (
            (upright and get_psi_from(stm,mirror_mode) == 0)
            or (not upright and get_psi_from(stm,mirror_mode) == pm90)
        ):
            dmatv = np.eye(3)
        else:
            dmatv = dmatvp if upright else dmatvm 

    # ------------------------------------------------------------------
    # 鉛直移動のための"stm"アップデート
    if effect_level in (2, 8):
        nstep1 = data['nstep1']  # ハイスピードモードのステップ数(Constant)
        stm = stm @ dmatv
        session["stm"] = stm.tolist()  # session <- stm
        session['needs_init'] = True

    # ------------------------------------------------------------------
    # 高速補正/高画質補正
    # params
    if effect_level in (2, 4, 6, 8, 7, 9):
        mode = "HS"
        nstep1 = data['nstep1']  # ハイスピードモードのステップ数(Constant)
        if session["needs_init"]:
            gbias = rdinit_sub(tcp1, stm, nstep1, twidth1, theight1, rhagv, gmp, gir)  # kokopoint
        session['needs_init'] = False
        hosei_sub_hs(ttupcd1, stupcd1, tcp1, stm, fast, nstep1, twidth1, theight1, params)
        timage = Image.fromarray(ttupcd1, 'RGB')
        # timage.thumbnail((twidth1/np.sqrt(2.), theight1/np.sqrt(2.)))
        # timage = timage.filter(ImageFilter.SHARPEN)
        # timage = timage.filter(ImageFilter.DETAIL)
        # save_path = "static/image.png"
        # timage.save(save_path)

    elif effect_level in (0, 1, 5):
        mode = 'HR'
        nstep2 = data['nstep2']  # ハイレゾモードのステップ数(Constant)
        session['needs_init'] = True
        gbias = rdinit_sub(tcp2, stm, nstep2, twidth2, theight2, rhagv, gmp, gir)  # kokopoint
        hosei_sub_hr(ttupcd2, stupcd2, tcp2, stm, fast, nstep2, twidth2, theight2, params)  # kokopoint
        timage = Image.fromarray(ttupcd2, 'RGB')
    
    if Footstep:
        print('=====', 'timage done', '=====', flush = True)
        save_path = "static/image.png"
        timage.save(save_path)

    # ------------------------------------------------------------------
    # 画像をバイナリデータとして送信
    img_io = io.BytesIO()
    timage.save(img_io, format="PNG")
    img_io.seek(0)
    if TimeMMs:
        print(f"Processing {template_key} completed in {time.time() - start_time:.2f} sec. {mode}")
        start_time = time.time()
        session["start_time"] = start_time
    return send_file(img_io, mimetype="image/png")

    # ------------------------------------------------------------------
    # staticに画像を保存
    if False:
        save_path = "static/image.png"
        timage.save(save_path)
        return jsonify({"image_url": "/" + save_path})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )


# Flaskのメインルーティング
@app.route('/page/<template_name>')
def dynamic_template(template_name):
    # 拡張子とテンプレート名を分ける
    key = Path(template_name).stem

    template_key = Path(template_name).stem
    try:
        if template_key not in preprocess_cache:
            if TimeMMs:
                start_time = time.time()
            preprocess_cache[template_key] = pre_process(template_key)
            if TimeMMs:
                print(f"Pre-Processing {template_key} completed in {time.time() - start_time:.2f} sec.")
                start_time = time.time()
                session["start_time"] = start_time
        return render_template(template_name)
    except ConfigError as e:
        return render_template('error.html', message=str(e), page_name=f"{template_key}.json")


@app.route('/')
def index():
    return render_template('index.html')
