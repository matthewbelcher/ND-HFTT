import os


def read_all(path='data/fomc_statements/'):
    files = os.listdir(path)
    for file in files:
        with open(path+file) as f:
            html_content = f.read()
            found_index = html_content.find('FOMC statement</h3>')
            if found_index == -1:
                print(f'did not find FOMC statement declaration in {file}')
                continue
            found_index_2 = html_content.find('<div class="col-xs-12 col-sm-8 col-md-8">', found_index)
            if found_index_2 == -1:
                print(f'did not find <div> declaration in {file}')
                continue
            start_index = found_index_2 + len('<div class="col-xs-12 col-sm-8 col-md-8">')
            end_index = html_content.find('</div>', start_index)
            if end_index == -1:
                print(f'did not find </div> declaration in {file}')
                continue


            statement = html_content[start_index:end_index]
            statement = statement.strip().replace('<p>', ' ').replace('</p>', ' ').replace('<P>', ' ').replace('</P>', ' ')
            ind = statement.find('<a href=')
            if ind != -1:    
                statement = statement[:ind]
            ind = statement.find('<a target=')
            if ind != -1:    
                statement = statement[:ind]
            print(file)
            print(statement)
            print()
            


if __name__ == "__main__":
    read_all()