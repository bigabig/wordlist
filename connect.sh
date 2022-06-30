#!/bin/bash

open_ssh_tunnels() {
  echo "Opening SSH tunnel from ltdwise:8501 ..."
  ssh -L "8501:127.0.0.1:8501" -f -N ltdwise
}

kill_ssh_pipes() {
  declare -a arr=("8501")

  for i in "${arr[@]}"
  do
    echo "Closing SSH tunnel from ltdwise:$i ..."
    pid=$(pgrep -f -a "ssh -L $i")
    kill "$pid"
  done
}

PS3="open or close the tunnels? "
select method in open kill; do
  case $method in
  open)
    open_ssh_tunnels
    break
    ;;
  kill)
    kill_ssh_pipes
    break
    ;;
  *)
    echo "Invalid option $REPLY"
    ;;
  esac
  echo
done
echo