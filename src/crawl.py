import numpy as np
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from time import sleep
import pandas as pd
from datetime import date
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CellPhones")

class WebDriverCustomOptions:
    def __init__(self):
        options = Options()
        # options.add_argument("--headless")  # Chế độ không có giao diện đồ họa
        # options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--no-sandbox")
        # options.add_argument("--incognito")  # Chế độ duyệt web riêng tư

        # Khởi tạo trình điều khiển WebDriver với các tùy chọn đã được thiết lập
        self.driver = webdriver.Firefox(options=options)
        
    @staticmethod
    def scrape_data(self):
        pass
    
class HeadPhoneCellPhones(WebDriverCustomOptions):
    def __init__(self):
        super().__init__() # Gọi phương thức khởi tạo của lớp cha để khởi tạo trình duyệt với các tùy chọn tùy chỉnh
        self.url = "https://cellphones.com.vn/thiet-bi-am-thanh/tai-nghe/tai-nghe-bluetooth.html?order=filter_price&dir=desc"
    
    def scrape_data(self):
        try:
            self.driver.get(self.url)
            
            # Hiển thị tất cả sản phẩm
            check_ = self.driver.find_element(By.XPATH, "//a[@class='button btn-show-more button__show-more-product']")
            x = check_.text.split(' ')[2]
            while x != '1':
                check_.click()
                x = check_.text.split(' ')[2]
                sleep(2)
            
            # Tên sản phẩm
            all_element_name_item = self.driver.find_elements(By.XPATH, "//div[@class='product__name']/h3")
            all_name_item = [name.text for name in all_element_name_item]
            logger.info("Tên sản phẩm: {}".format(len(all_name_item)))
            
            # Giá tiền
            info_price = self.driver.find_elements(By.XPATH, "//div[@class='box-info__box-price']")
            price_show = [] # giá tiền hiển thị
            price_through = []  # giá tiền còn
            price_sale = []     # giá tiền giảm
            for info_ in info_price:
                check = info_.find_elements(By.TAG_NAME, "p")
                if len(check) == 1:
                    show = check[0].text
                    price_show.append(show)
                    price_through.append('0')
                    price_sale.append('0')
                else:
                    show = check[0].text
                    through = check[1].text
                    sale = check[2].text
                    price_show.append(show)
                    price_through.append(through)
                    price_sale.append(sale)
            logger.info("Giá tiền hiển thị: {}".format(len(price_show)))
            logger.info("Giá tiền còn lại: {}".format(len(price_through)))
            logger.info("Giá tiền giảm: {}".format(len(price_sale)))
            
            # Hình ảnh
            all_element_url_img = self.driver.find_elements(By.XPATH, "//img[@class='product__img']")
            all_url_img = [url_img.get_attribute('src') for url_img in all_element_url_img]
            logger.info("Đường dẫn hình ảnh: {}".format(len(all_url_img)))
            
            # Link sản phẩm
            all_element_link_item = self.driver.find_elements(By.XPATH, "//a[@class='product__link button__link']")
            all_link_item = [link.get_attribute("href") for link in all_element_link_item]
            logger.info("Đường dẫn sản phẩm: {}".format(len(all_link_item)))
            
            # Thông tin chi tiết
            all_element_link_item = self.driver.find_elements(By.XPATH, "//a[@class='product__link button__link']")
            all_link_item = [link.get_attribute("href") for link in all_element_link_item] # Link chi tiết sản phẩm
            arr_all_info_item = []
            all_special_info = []
            for link_item in all_link_item:
                arr_ = []
                self.driver.get(link_item)
                sleep(np.random.randint(1,5))
                all_info_item = self.driver.find_elements(By.XPATH, "//div[@class='item-warranty-info']//div[@class='description']")
                for item in all_info_item:
                    arr_.append(item.text)
                
                info_item = ",".join(arr_)
                arr_all_info_item.append(str(info_item))
                sleep(1)
                self.driver.find_element(By.XPATH, "//a[@class='btn-show-more button__content-show-more']").click()
                sleep(2)
                special_info = self.driver.find_element(By.XPATH, "//div[@class='cps-block-content']").text
                all_special_info.append(special_info.replace('\n', ' '))
            logger.info("Thông tin chi tiết sản phẩm: {}".format(len(all_special_info)))
            
            
            df = pd.DataFrame({'Tên sản phẩm': all_name_item,
                                'Gía tiền': price_show,
                                'Gía tiền còn lại': price_through,
                                'Giá tiền giảm': price_sale,
                                'Thông tin chi tiết':all_special_info,
                                'Hình ảnh': all_url_img,
                                'Link': all_link_item})
            df.to_csv("../data/cellphone.csv", index=False)
            logger.info("Dữ liệu về tai nghe Cellphones đã XONG.")
            logger.info(df.shape[0])
            
            # return df
        except Exception as e:
            raise e
        finally:
            self.driver.quit()
            
    
class HeadPhoneTiki(WebDriverCustomOptions):
    def __init__(self):
        super().__init__() # Gọi phương thức khởi tạo của lớp cha để khởi tạo trình duyệt với các tùy chọn tùy chỉnh
        # self.url = 
            
if __name__ == "__main__":
    cellphone = HeadPhoneCellPhones()
    cellphone.scrape_data()
    
    # tiki = HeadPhoneTiki()
    # def scrape_data(self):
    #     try:
            
            
    #     except Exception as e:
    #         raise e
    #     finally:
    #         self.driver.quit()
        