using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace Pentago
{
    public partial class Form1 : Form
    {
        ActionMove action = new ActionMove();
        StateSpace stateMagr = new StateSpace();

        List<System.Windows.Forms.Button> buttons;
        List<System.Windows.Forms.Button> rotateButtons;

        public Form1()
        {
            ANN n = new ANN();

            Vector<double> input = Vector<double>.Build.Random(2);

            n.TestDerrivatives(input);

            InitializeComponent();

            buttons = new List<Button> 
            {
                button1,
                button2,
                button3,
                button4,
                button5,
                button6,
                button7,
                button8,
                button9,
                button10,
                button11,
                button12,
                button13,
                button14,
                button15,
                button16,
                button17,
                button18,
                button_19,
                button_20,
                button_21,
                button_22,
                button_23,
                button_24,
                button_25,
                button_26,
                button_27,
                button_28,
                button_29,
                button_30,
                button_31,
                button_32,
                button_33,
                button_34,
                button_35,
                button_36,
            };


            rotateButtons = new List<Button> 
            {
                button19,
                button20,
                button24,
                button23,
                button21,
                button22,
                button25,
                button26};

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        void GameOver()
        {
            MessageBox.Show("gAME OVER");
            stateMagr.ResetState();
            buttons.ForEach(x => { x.Text = ""; x.Enabled = true; });
            rotateButtons.ForEach(x => x.Enabled = true);
            buttonPlay.Enabled = true;
        }

        private void AcquireMove()
        {
            if (stateMagr.AcquireMove(action, buttons))
            {
                GameOver();
            }

            //rotateButtons.ForEach(x => x.Enabled = false);
            buttonPlay.Enabled = true;
        }
        

        private void button17_Click(object sender, EventArgs e)
        {
            stateMagr.SelfTrain2(buttons,100000);
            stateMagr.ResetState();
        }

        private void button18_Click(object sender, EventArgs e)
        {
            ActionMove action = new ActionMove();
            if (stateMagr.PredictMove(buttons, ref action, 0))
            {
                GameOver();
                return;
            }
            label2.Text = action.Value.ToString();

            rotateButtons.ForEach(x => x.Enabled = true);
            buttonPlay.Enabled = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            button1.Text = "X";
            action.PositionsId = 0;
        }

        private void button19_Click(object sender, EventArgs e)
        {
            action.Rotation = 2;
            AcquireMove();
        }

        private void button20_Click(object sender, EventArgs e)
        {
            action.Rotation = 1;
            AcquireMove();
        }

        private void button24_Click(object sender, EventArgs e)
        {
            action.Rotation = 3;
            AcquireMove();
        }

        private void button23_Click(object sender, EventArgs e)
        {
            action.Rotation = 4;
            AcquireMove();
        }

        private void button21_Click(object sender, EventArgs e)
        {
            action.Rotation = 6;
            AcquireMove();
        }

        private void button22_Click(object sender, EventArgs e)
        {
            action.Rotation = 5;
            AcquireMove();
        }

        private void button26_Click(object sender, EventArgs e)
        {
            action.Rotation = 7;
            AcquireMove();
        }

        private void button25_Click(object sender, EventArgs e)
        {
            action.Rotation = 8;
            AcquireMove();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            button2.Text = "X";
            action.PositionsId = 1;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            button3.Text = "X";
            action.PositionsId = 2;
        }

        private void button4_Click(object sender, EventArgs e)
        {
            button4.Text = "X";
            action.PositionsId = 3;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            button5.Text = "X";
            action.PositionsId = 4;
        }

        private void button6_Click(object sender, EventArgs e)
        {
            button6.Text = "X";
            action.PositionsId = 5;
        }

        private void button7_Click(object sender, EventArgs e)
        {
            button7.Text = "X";
            action.PositionsId = 6;
        }

        private void button8_Click(object sender, EventArgs e)
        {
            button8.Text = "X";
            action.PositionsId = 7;
        }

        private void button9_Click(object sender, EventArgs e)
        {
            button9.Text = "X";
            action.PositionsId = 8;
        }

        private void button10_Click(object sender, EventArgs e)
        {
            button10.Text = "X";
            action.PositionsId = 9;
        }

        private void button11_Click(object sender, EventArgs e)
        {
            button11.Text = "X";
            action.PositionsId = 10;
        }

        private void button12_Click(object sender, EventArgs e)
        {
            button12.Text = "X";
            action.PositionsId = 11;
        }

        private void button13_Click(object sender, EventArgs e)
        {
            button13.Text = "X";
            action.PositionsId = 12;
        }

        private void button14_Click(object sender, EventArgs e)
        {
            button14.Text = "X";
            action.PositionsId = 13;
        }

        private void button15_Click(object sender, EventArgs e)
        {
            button15.Text = "X";
            action.PositionsId = 14;
        }

        private void button16_Click(object sender, EventArgs e)
        {
            button16.Text = "X";
            action.PositionsId = 15;
        }

        private void button27_Click(object sender, EventArgs e)
        {
            SaveFileDialog openFileDialog1 = new SaveFileDialog();
            openFileDialog1.Filter = "All files (*.*)|*.*";
            openFileDialog1.FilterIndex = 1;
 
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                using (System.IO.FileStream fs = File.OpenWrite(openFileDialog1.FileName))
                {
                    using (var writer = new StreamWriter(fs))
                    {
                        stateMagr.Write(writer);
                    }
                }
            }
        }

        private void button28_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog1 = new OpenFileDialog();
            openFileDialog1.Filter = "All files (*.*)|*.*";
            openFileDialog1.FilterIndex = 2;

            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                using (System.IO.FileStream fs = File.OpenRead(openFileDialog1.FileName))
                {
                    using (var reader = new StreamReader(fs))
                    {
                        stateMagr.Load(reader);
                    }
                }
            }
        }

        private void button29_Click(object sender, EventArgs e)
        {
            stateMagr.LearnQNetwork();
        }

        private void button17_Click_1(object sender, EventArgs e)
        {
            button17.Text = "X";
            action.PositionsId = 16;
        }

        private void button18_Click_1(object sender, EventArgs e)
        {
            button18.Text = "X";
            action.PositionsId = 17;
        }

        private void button_19_Click(object sender, EventArgs e)
        {
            button_19.Text = "X";
            action.PositionsId = 18;
        }

        private void button_20_Click(object sender, EventArgs e)
        {
            button_20.Text = "X";
            action.PositionsId = 19;
        }

        private void button_21_Click(object sender, EventArgs e)
        {
            button_21.Text = "X";
            action.PositionsId = 20;
        }

        private void button_22_Click(object sender, EventArgs e)
        {
            button_22.Text = "X";
            action.PositionsId = 21;
        }

        private void button_23_Click(object sender, EventArgs e)
        {
            button_23.Text = "X";
            action.PositionsId = 22;
        }

        private void button_24_Click(object sender, EventArgs e)
        {
            button_24.Text = "X";
            action.PositionsId = 23;
        }

        private void button_25_Click(object sender, EventArgs e)
        {
            button_25.Text = "X";
            action.PositionsId = 24;
        }

        private void button_26_Click(object sender, EventArgs e)
        {
            button_26.Text = "X";
            action.PositionsId = 25;
        }

        private void button_27_Click(object sender, EventArgs e)
        {
            button_27.Text = "X";
            action.PositionsId = 26;
        }

        private void button_28_Click(object sender, EventArgs e)
        {
            button_28.Text = "X";
            action.PositionsId = 27;
        }

        private void button_29_Click(object sender, EventArgs e)
        {
            button_29.Text = "X";
            action.PositionsId = 28;
        }

        private void button_30_Click(object sender, EventArgs e)
        {
            button_30.Text = "X";
            action.PositionsId = 29;
        }

        private void button_31_Click(object sender, EventArgs e)
        {
            button_31.Text = "X";
            action.PositionsId = 30;
        }

        private void button_32_Click(object sender, EventArgs e)
        {
            button_32.Text = "X";
            action.PositionsId = 31;
        }

        private void button_33_Click(object sender, EventArgs e)
        {
            button_33.Text = "X";
            action.PositionsId = 32;
        }

        private void button_34_Click(object sender, EventArgs e)
        {
            button_34.Text = "X";
            action.PositionsId = 33;
        }

        private void button_35_Click(object sender, EventArgs e)
        {
            button_35.Text = "X";
            action.PositionsId = 34;
        }

        private void button_36_Click(object sender, EventArgs e)
        {
            button_36.Text = "X";
            action.PositionsId = 35;
        }
    }
}
