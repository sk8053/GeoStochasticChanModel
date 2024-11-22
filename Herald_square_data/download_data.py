import gdown
import os

root_path = os.path.dirname(os.path.abspath(__file__))#.split('/')[0]

def download_data(pre_processed_only = False):
    if pre_processed_only is True:
        url_processed_data ={'data_total':'https://drive.google.com/file/d/16ezCkaa2F1QXK8p_QlYSxgrLB2VtgUvj/view?usp=drive_link',
                            'cond_total': 'https://drive.google.com/file/d/1YScuPDPhaYTr7ed21kNeS74dSjP02Bot/view?usp=drive_link'
                            }
        for d in url_processed_data.keys():
            output_path = os.path.join(root_path, f'{d}.pickle')
            if os.path.isfile(output_path) is False:
                print(f'++++++++++++++++++++++ Download csv files for {d} from google drive ++++++++++++++ ')
                url = url_processed_data[d]
                gdown.download(url, output_path, quiet=False,fuzzy=True)
    
    else:
        url_processed_data ={'data_total':'https://drive.google.com/file/d/16ezCkaa2F1QXK8p_QlYSxgrLB2VtgUvj/view?usp=drive_link',
                            'cond_total': 'https://drive.google.com/file/d/1YScuPDPhaYTr7ed21kNeS74dSjP02Bot/view?usp=drive_link'
                            }
        for d in url_processed_data.keys():
            output_path = os.path.join(root_path, d)
            if os.path.isfile(output_path) is False:
                print(f'++++++++++++++++++++++ Download csv files for {d} from google drive ++++++++++++++ ')
                url = url_processed_data[d]
                gdown.download(url, output_path, quiet=False,fuzzy=True)
        
        url_map= {1.6:'https://drive.google.com/file/d/1ruPwB9ovEYhENSVIghpWRcqqX1SqBdSW/view?usp=drive_link',
            30: 'https://drive.google.com/file/d/1cRdI2ssR5M9H2oFvCSYXZKegG2scp9yS/view?usp=drive_link',
            60: 'https://drive.google.com/file/d/19eIunR1EasAHd1_7cekTtZ0lTdSUhwAE/view?usp=drive_link',
            90: 'https://drive.google.com/file/d/1hVVcQSsRKD6jnUzYFIKFtDAB0MsRaNlW/view?usp=drive_link',
            120:'https://drive.google.com/file/d/1YzYC60_9w2m9dSfcc8UivIbcbQ9dxUO3/view?usp=drive_link'
            }

        for height in [1.6, 30, 60, 90, 120]:
            output_path = os.path.join(root_path, f'bs_{height}m.csv')
            if os.path.isfile(output_path) is False:
                print(f'++++++++++++++++++++++ Download csv files at height = {height} from google drive ++++++++++++++ ')
                url = url_map[height]
                gdown.download(url, output_path, quiet=False,fuzzy=True)
        
        
