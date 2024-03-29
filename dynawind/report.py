from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
import itertools
import os
class Report(object):
    def __init__(self, path, filename, title, author, client, project,
                 project_code,location = 'nA', version=None, document_nr=1,
                 document_type = "AR"):
        self.path = path
        self.filename = filename
        self.author = author
        self.title = title
        self.document_type = document_type
        self.client = client
        self.location = location
        self.project = project
        self.project_code = project_code
        self.document_nr = document_nr
        if version is None:
            self.version = 1
        self.body = []

        self.add_header()

    def add_header(self):
        self.write("\\documentclass{24SEA_Report_Template}\n")
        self.write("\\FileReference{"+self.document_type+"}{"+str(self.document_nr).zfill(3)+"}\\par\n")
        self.write("\\telephone{0032/2629 23 90}\n")
        self.write("\\Client{"+self.client+"}\n")
        self.write("\\ClientReference{...}\n")
        self.write("\\newcommand{\\turb}{"+self.location+"}\n")

        self.write("\\Version{"+str(self.version)+"}\n")
        self.write("\\ProjectCode{"+str(self.project_code).zfill(4)+"}\n")
        self.write("\\ReportDate{\\today}\n")
        self.write("\\author{"+self.author+"}\n")
        self.write("\\title{"+self.title+"}\n")
        self.write("\\begin{document}\n")

    def add_table(self, table):
        self.write('\\begin{table}\n')
        self.write("\\centering\n")
        self.write(table.write())
        self.add_caption(table.caption_str)
        self.write('\\end{table}\n')

    def add_caption(self, caption_str):
        if caption_str is not None:
            self.write("\caption{"+caption_str+"}\n")


    def add_figure(self, figure, sideways=False, width = 0.8):
        """ Adds a Figure object to the report
        Takes several inputs associated with the LaTeX layout:
            - width : if <1, this is relative to textwidth, else in cm
        """
        figure_path = os.path.join(self.path, "Figures")

        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        figure_path = os.path.join(figure_path, figure.filename + '.png' )
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylabel(plt.gca().get_ylabel(),fontsize=20)
        plt.xlabel(plt.gca().get_xlabel(),fontsize=20)
        
        figure.save(figure_path)

        self.write("\\begin{figure}\n")
        self.write("\\centering\n")
        includegraphics_settings = []
        if width < 1:
            includegraphics_settings.append("width="+str(width)+"\\textwidth")
        else:
            includegraphics_settings.append("width="+str(width)+"cm")
        figure_str = "\\includegraphics["

        for setting in includegraphics_settings:
            figure_str = figure_str + setting
        figure_str = figure_str + "]{./Figures/"+ figure.filename+'.png' + "}\n"
        self.write(figure_str)
        self.add_caption(figure.caption_str)
        self.write("\\end{figure}\n")
        pass

    def compile_report(self,
                       template_folder=r"\\192.168.119.12\Templates\Documents\24SEA\LaTeX_Report_Template",
                       remove_aux=False,
                       remove_figures=False):
        from shutil import copy2, rmtree
        from subprocess import call

        if not os.path.isdir(template_folder):
            raise ImportError("Template file not found/reachable")
        copy2(os.path.join(template_folder,'24SEA_Report_Template.cls'),
              self.path)
        copy2(os.path.join(template_folder,'24SEA_logo.png'),
              self.path)

        self.export_tex()
        print('Compiling ...')
        call("pdflatex "+self.filename+'.tex',
              cwd=os.path.realpath(self.path))
        # Because pdflatex requires two runs to render references
        print('Rendering references ...')
        call("pdflatex "+self.filename+'.tex',
              cwd=os.path.realpath(self.path))
        if remove_aux:
            # Remove template files
            os.remove(os.path.join(self.path,'24SEA_Report_Template.cls'))
            os.remove(os.path.join(self.path,'24SEA_logo.png'))
            # Remove aux files generated by LaTeX
            auxs = ['.aux','.log','.tex','.out']
            for aux in auxs:
                os.remove(os.path.join(self.path,self.filename+aux))


        if remove_figures:
            # Remove the figures Folder
            if os.path.exists(os.path.join(self.path, "Figures")):
                rmtree(os.path.join(self.path, "Figures"), ignore_errors=True)


    def export_tex(self):
        self.write("\\end{document}\n")
        with open(os.path.join(self.path, self.filename +".tex"), "w") as f:
            for line in self.body:
                f.write(line)

    def write(self, string):
        self.body.extend(string)


class Table(object):
    """ Table class object to add in reports """
    def __init__(self, data, no_columns=4):
        # data passed to the Table object is a list of dicts
        self.data = data
        self.caption_str = None
        self.no_columns = no_columns
        
    def caption(self, caption):
        self.caption_str = caption

    def write(self, style='Stats'):
        # write the table depending the style
        strings = []
        if style == 'Stats':
            table_columns = 'lc'
            for i in range(1,int(self.no_columns/2)):
                table_columns += '|lc'
            strings.append('\\begin{tabular}{'+table_columns+'}\n')
            col_ind=2
            for item in self.data:
                string = '\\bf{'+item['string']+':}& '

                if isinstance(item['value'],str):
                    string = string + item['value']
                else:
                    string = string + '{:.2f}'.format(item['value'])

                if col_ind == self.no_columns:
                    string = '&' + string + "\\\\\n"
                    col_ind = 0
                col_ind +=2
                strings.append(string)
            strings.append('\\end{tabular}\n')
        return strings



class Figure(object):
    """ Figure class object to add in reports """

    newid = itertools.count()
    def __init__(self,name = None, dpi=120, figsize=(20,4), caption = None):
        self.caption_str = caption
        self.dpi = dpi
        self.fig_handle = plt.figure(figsize=figsize, dpi=dpi)
        if name is None:
            self.filename = 'DYNAwind_fig_'+str(next(Figure.newid)).zfill(3)

    def caption(self, caption):
        self.caption_str = caption

    def close(self):
        plt.close(self.fig_handle)

    def save(self, path):
        savefig(path, dpi=self.dpi, bbox_inches='tight')
