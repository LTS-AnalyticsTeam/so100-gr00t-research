import pytest
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vla_auto_recover.processing.utils.image_convert import (
    imgmsg_to_ndarray, 
    numpy_to_imgmsg,
    DTYPE_CHANNEL2ENCODING
)


class TestImageConvert:
    """画像変換関数のテストクラス"""
    
    def test_mono8_conversion_roundtrip(self):
        """mono8エンコーディングの変換と逆変換テスト"""
        # テスト用のグレースケール画像を作成
        original_array = np.random.randint(0, 256, size=(100, 150), dtype=np.uint8)
        
        # numpy → Image
        img_msg = numpy_to_imgmsg(original_array, encoding='mono8')
        
        # 基本的なメッセージプロパティを確認
        assert img_msg.height == 100
        assert img_msg.width == 150
        assert img_msg.encoding == 'mono8'
        assert img_msg.step == 150  # width * channels * itemsize
        
        # Image → numpy
        converted_array = imgmsg_to_ndarray(img_msg)
        
        # 元の配列と変換後の配列が同じであることを確認
        np.testing.assert_array_equal(original_array, converted_array)
        assert converted_array.shape == (100, 150)
        assert converted_array.dtype == np.uint8

    def test_bgr8_conversion_roundtrip(self):
        """bgr8エンコーディングの変換と逆変換テスト"""
        # テスト用のBGR画像を作成
        original_array = np.random.randint(0, 256, size=(80, 120, 3), dtype=np.uint8)
        
        # numpy → Image
        img_msg = numpy_to_imgmsg(original_array, encoding='bgr8')
        
        # 基本的なメッセージプロパティを確認
        assert img_msg.height == 80
        assert img_msg.width == 120
        assert img_msg.encoding == 'bgr8'
        assert img_msg.step == 360  # 120 * 3 * 1
        
        # Image → numpy
        converted_array = imgmsg_to_ndarray(img_msg)
        
        # 元の配列と変換後の配列が同じであることを確認
        np.testing.assert_array_equal(original_array, converted_array)
        assert converted_array.shape == (80, 120, 3)
        assert converted_array.dtype == np.uint8

    def test_rgb8_conversion_roundtrip(self):
        """rgb8エンコーディングの変換と逆変換テスト"""
        # テスト用のRGB画像を作成
        original_array = np.random.randint(0, 256, size=(60, 90, 3), dtype=np.uint8)
        
        # numpy → Image
        img_msg = numpy_to_imgmsg(original_array, encoding='rgb8')
        
        # 基本的なメッセージプロパティを確認
        assert img_msg.height == 60
        assert img_msg.width == 90
        assert img_msg.encoding == 'rgb8'
        
        # Image → numpy
        converted_array = imgmsg_to_ndarray(img_msg)
        
        # 元の配列と変換後の配列が同じであることを確認
        np.testing.assert_array_equal(original_array, converted_array)
        assert converted_array.shape == (60, 90, 3)

    def test_mono16_conversion_roundtrip(self):
        """mono16エンコーディングの変換と逆変換テスト"""
        # テスト用の16bitグレースケール画像を作成
        original_array = np.random.randint(0, 65536, size=(50, 75), dtype=np.uint16)
        
        # numpy → Image
        img_msg = numpy_to_imgmsg(original_array, encoding='mono16')
        
        # 基本的なメッセージプロパティを確認
        assert img_msg.height == 50
        assert img_msg.width == 75
        assert img_msg.encoding == 'mono16'
        assert img_msg.step == 150  # 75 * 1 * 2
        
        # Image → numpy
        converted_array = imgmsg_to_ndarray(img_msg)
        
        # 元の配列と変換後の配列が同じであることを確認
        np.testing.assert_array_equal(original_array, converted_array)
        assert converted_array.shape == (50, 75)
        assert converted_array.dtype == np.uint16

    def test_auto_encoding_detection(self):
        """エンコーディング自動判定のテスト"""
        # mono8の自動判定
        mono_array = np.random.randint(0, 256, size=(40, 60), dtype=np.uint8)
        img_msg = numpy_to_imgmsg(mono_array)  # encoding=None
        assert img_msg.encoding == 'mono8'
        
        # bgr8の自動判定
        bgr_array = np.random.randint(0, 256, size=(40, 60, 3), dtype=np.uint8)
        img_msg = numpy_to_imgmsg(bgr_array)  # encoding=None
        assert img_msg.encoding == 'bgr8'
        
        # mono16の自動判定
        mono16_array = np.random.randint(0, 65536, size=(40, 60), dtype=np.uint16)
        img_msg = numpy_to_imgmsg(mono16_array)  # encoding=None
        assert img_msg.encoding == 'mono16'

    def test_header_parameters(self):
        """ヘッダーパラメータのテスト"""
        test_array = np.random.randint(0, 256, size=(30, 40), dtype=np.uint8)
        frame_id = 'camera_frame'
        
        # フレームIDを指定して変換
        img_msg = numpy_to_imgmsg(test_array, frame_id=frame_id)
        assert img_msg.header.frame_id == frame_id

    def test_non_contiguous_array(self):
        """非連続配列の処理テスト"""
        # 非連続配列を作成（スライスで作成）
        large_array = np.random.randint(0, 256, size=(100, 200, 3), dtype=np.uint8)
        non_contiguous = large_array[::2, ::2, :]  # 非連続になる
        
        assert not non_contiguous.flags['C_CONTIGUOUS']
        
        # 変換が正常に動作することを確認
        img_msg = numpy_to_imgmsg(non_contiguous)
        converted_back = imgmsg_to_ndarray(img_msg)
        
        np.testing.assert_array_equal(non_contiguous, converted_back)

    def test_unsupported_encoding_error(self):
        """サポートされていないエンコーディングのエラーテスト"""
        # 無効なエンコーディングでImageメッセージを作成
        img_msg = Image()
        img_msg.encoding = 'unsupported_encoding'
        img_msg.height = 10
        img_msg.width = 10
        img_msg.data = b'\x00' * 100
        
        with pytest.raises(TypeError, match="Unsupported encoding"):
            imgmsg_to_ndarray(img_msg)

    def test_invalid_array_shape_error(self):
        """無効な配列形状のエラーテスト"""
        # 4次元配列（無効）
        invalid_array = np.random.randint(0, 256, size=(10, 20, 30, 4), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="arr must have shape"):
            numpy_to_imgmsg(invalid_array)

    def test_unsupported_dtype_channels_error(self):
        """サポートされていないdtype/channels組み合わせのエラーテスト"""
        # float32のような未サポートdtype
        unsupported_array = np.random.rand(30, 40).astype(np.float32)
        
        with pytest.raises(ValueError, match="encoding を自動判定できません"):
            numpy_to_imgmsg(unsupported_array)

    def test_bigendian_handling(self):
        """ビッグエンディアンの処理テスト"""
        original_array = np.random.randint(0, 65536, size=(20, 30), dtype=np.uint16)
        
        # リトルエンディアンで変換
        img_msg = numpy_to_imgmsg(original_array)
        assert img_msg.is_bigendian == 0
        
        # 手動でビッグエンディアンに設定
        img_msg.is_bigendian = 1
        
        # 変換が正常に動作することを確認
        converted_array = imgmsg_to_ndarray(img_msg)
        # ビッグエンディアンの場合、バイト順序が変わるのでdtypeが異なる
        assert converted_array.shape == original_array.shape
        assert converted_array.dtype.kind == 'u'  # unsigned integer
        assert converted_array.dtype.itemsize == 2  # 2 bytes

    def test_rgba8_with_explicit_encoding(self):
        """明示的にrgba8エンコーディングを指定するテスト"""
        # RGBA画像を作成
        rgba_array = np.random.randint(0, 256, size=(25, 35, 4), dtype=np.uint8)
        
        # 明示的にrgba8を指定
        img_msg = numpy_to_imgmsg(rgba_array, encoding='rgba8')
        assert img_msg.encoding == 'rgba8'
        assert img_msg.step == 140  # 35 * 4 * 1
        
        # 変換確認（ただし、imgmsg_to_ndarrayはrgba8をサポートしていないので例外が発生するはず）
        with pytest.raises(TypeError, match="Unsupported encoding"):
            imgmsg_to_ndarray(img_msg)

    def test_edge_case_small_image(self):
        """極小画像のエッジケーステスト"""
        # 1x1ピクセルの画像
        tiny_array = np.array([[255]], dtype=np.uint8)
        
        img_msg = numpy_to_imgmsg(tiny_array)
        converted_array = imgmsg_to_ndarray(img_msg)
        
        np.testing.assert_array_equal(tiny_array, converted_array)
        assert converted_array.shape == (1, 1)

    def test_different_array_values(self):
        """異なる値を持つ配列での変換テスト"""
        # 最小値・最大値を含む配列
        test_values = [
            np.zeros((10, 15), dtype=np.uint8),  # 全て0
            np.full((10, 15), 255, dtype=np.uint8),  # 全て255
            np.arange(150, dtype=np.uint8).reshape(10, 15),  # 連続値
        ]
        
        for test_array in test_values:
            img_msg = numpy_to_imgmsg(test_array)
            converted_array = imgmsg_to_ndarray(img_msg)
            np.testing.assert_array_equal(test_array, converted_array)