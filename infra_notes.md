# GCP Infrastructure Notes

## Create a Jupyter Notebook Deep Learning Instance
Follow the instructions [here](https://cloud.google.com/ai-platform/notebooks/docs/create-new)
- NB: NVIDIA T4 GPU cards are the cheapest and also offer great performance.

### Configure your instance so you can connect over SSH from VS Code
[Create SSH key](https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys#createsshkeys)
```
KEY_FILENAME=<instance name>
USERNAME=todd_morrill
ssh-keygen -t rsa -f ~/.ssh/$KEY_FILENAME -C $USERNAME
chmod 400 ~/.ssh/$KEY_FILENAME
```
[Add your key to your VM instance](https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys#instance-only)
```
# for example, copy and paste the results of running this command
cat /Users/tmorrill002/.ssh/<instance name>.pub
```
Test your connection (first retrieve the VM's IP address)

If you don't have gcloud installed, follow [these instructions](https://cloud.google.com/sdk/docs/install).
```
INSTANCE_NAME=<instance name>
ZONE=us-central1-a
gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
ssh -i ~/.ssh/<instance name> todd_morrill@<IP address>
```
Connect to your VM over [ssh in VS Code](https://code.visualstudio.com/docs/remote/ssh#_remember-hosts-and-advanced-settings) by:
- typing cmd+shift+p to open the command palette
- searching for Remote-SSH: Add New SSH Host
    - NB: install the Remote - SSH extension if you don't already have it
- copy and paste the same command you used above to test the connection.
- your user folder to start coding (e.g. /home/todd_morrill)
Install all the extensions you would typically use (on the server)
- Python, Intellicode, Python Docstring Generator, etc.

Connect to your VM's Jupyter Notebook
- Simply click "Open JupyterLab" on GCP under AI Platform Notebooks

### Set up Github access **on your VM**
Detailed instructions [here](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
```
ssh-keygen -t rsa -b 4096 -f ~/.ssh/todds-github -C "<email-address>"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/todds-github
cat ~/.ssh/todds-github.pub
# add this public key to github
# permanently add this key to your ssh config file
# https://stackoverflow.com/questions/3466626/how-to-permanently-add-a-private-key-with-ssh-add-on-ubuntu
echo "IdentityFile ~/.ssh/todds-github" >> ~/.ssh/config

git clone <your-repo>
```

### Set up your virtual environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```